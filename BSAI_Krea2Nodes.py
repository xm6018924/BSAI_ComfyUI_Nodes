"""
BSAI Krea 2 风格参考生图节点组（Krea 3 语义对齐版）

提供 BSAI Krea 2 Style Reference 和 BSAI Krea 2 Image 两个节点：
- BSAI Krea 2 Style Reference: 将参考图通过 Krea2 CLIP vision 编码为风格条件
- BSAI Krea 2 Image: 一体化生图节点（模型+提示词+最多8个风格参考+采样+VAE解码）

升级说明（ComfyUI 0.41 对齐）：
  官方 Krea 3 Image 支持"一个节点内文生图 + 局部编辑、最多 8 张参考图"。
  本节点将内置风格参考从 2 路扩展为 8 路（style_image_1~8 + 各自 strength/level），
  与官方 Krea 3 的 8 张参考图语义对齐；原有 1/2 路签名不变，旧工作流完全兼容。

style_reference 端口兼容：
  • BSAI Krea 2 Style Reference 的 CONDITIONING 输出
  • 官方 Krea 2 Style Reference 的 KREA_STYLE_REF 输出（自动下载 URL 图片并本地编码）

所有模型路径均通过 folder_paths 使用相对路径。
"""

import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.clip_vision
import comfy.model_management
import folder_paths
import torch
import math
import io
import logging

logger = logging.getLogger(__name__)

# 最多 8 路风格参考（对齐官方 Krea 3 Image 的 8 张参考图上限）
KREA_MAX_STYLE_IMAGES = 8

# ============================================================================
# Helper: 从 URL 下载图片为 tensor
# ============================================================================

def _download_image_from_url(url: str):
    """Download an image from an http(s) URL and return a ComfyUI IMAGE tensor [B, H, W, C] (0~1 float32).

    SSRF guard: only http/https schemes are allowed; file:// and other schemes are
    rejected. The host must be a public/hostname value (loopback/private addresses
    are blocked to prevent internal network reachability from a workflow-supplied URL).
    """
    try:
        import urllib.request
        from urllib.parse import urlparse
        from PIL import Image

        if not isinstance(url, str) or not url:
            logger.warning(f"[BSAI Krea2] Empty or invalid style reference URL")
            return None

        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            logger.warning(f"[BSAI Krea2] Blocked non-http(s) scheme in style reference URL: {scheme!r} ({url})")
            return None

        host = (parsed.hostname or "").lower()
        if not host:
            logger.warning(f"[BSAI Krea2] Style reference URL has no host: {url}")
            return None

        # Block loopback / private / link-local hosts to mitigate SSRF.
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            logger.warning(f"[BSAI Krea2] Blocked loopback host in style reference URL: {host}")
            return None
        if host.startswith(("10.", "192.168.", "169.254.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.")):
            logger.warning(f"[BSAI Krea2] Blocked private/link-local host in style reference URL: {host}")
            return None

        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-BSAI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize((img.width, img.height), Image.LANCZOS)
        arr = torch.from_numpy(
            (torch.tensor([list(img.getdata())]).reshape(img.height, img.width, 3).numpy() / 255.0)
        ).float()
        return arr.unsqueeze(0)  # [1, H, W, 3]
    except Exception as e:
        logger.warning(f"[BSAI Krea2] Failed to download style reference image URL: {url}, error: {e}")
        return None


# ============================================================================
# Helper: 缩放参考图到 vision encoder 分辨率
# ============================================================================

def _resize_image_for_vision(style_image, vision_resolution=384):
    samples = style_image.movedim(-1, 1)
    total_pixels = max(1, int(vision_resolution) * int(vision_resolution))
    source_pixels = max(1, int(samples.shape[2]) * int(samples.shape[3]))
    scale_by = math.sqrt(total_pixels / source_pixels)
    width = max(1, round(samples.shape[3] * scale_by))
    height = max(1, round(samples.shape[2] * scale_by))
    try:
        samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    except Exception:
        import torch.nn.functional as F
        samples = F.interpolate(samples, size=(height, width), mode="area")
    return samples.movedim(1, -1)[:, :, :, :3]


# ============================================================================
# Helper: 判断 style_reference 数据类型
# ============================================================================

def _is_krea_style_ref(data):
    """判断是否为官方 Krea 2 Style Reference 的 KREA_STYLE_REF 格式（list[dict]）。"""
    if not isinstance(data, list) or len(data) == 0:
        return False
    first = data[0]
    return isinstance(first, dict) and "url" in first and "strength" in first


def _is_conditioning(data):
    """判断是否为 ComfyUI 标准 CONDITIONING 格式。"""
    if not isinstance(data, list) or len(data) == 0:
        return False
    first = data[0]
    return (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and isinstance(first[0], torch.Tensor)
        and isinstance(first[1], dict)
    )


# ============================================================================
# Helper: 将 KREA_STYLE_REF 转为本地 CONDITIONING
# ============================================================================

def _krea_style_ref_to_conditioning(clip, krea_style_ref_list):
    """将官方 Krea Style Reference 的 URL 列表下载并编码为本地 CONDITIONING。"""
    conds = []
    for item in krea_style_ref_list:
        url = item.get("url", "")
        strength = float(item.get("strength", 1.0))
        if not url:
            continue
        img_tensor = _download_image_from_url(url)
        if img_tensor is None:
            continue
        resized = _resize_image_for_vision(img_tensor, 384)
        text = (
            "Use the attached reference image as a style guide. Transfer its visual language, "
            "color palette, material texture, brushwork, lighting, composition rhythm, and "
            "overall mood. Do not copy the depicted subject unless the target prompt asks for it.\n\n"
            "[Style Reference Image]"
        )
        tokens = clip.tokenize(text, images=[resized])
        cond = clip.encode_from_tokens_scheduled(tokens)
        if strength != 1.0:
            cond = [
                [c * strength, extra] if isinstance(c, torch.Tensor) else [c, extra]
                for c, extra in cond
            ]
        conds.extend(cond)
    return conds


# ============================================================================
# Shared: 风格语义指令
# ============================================================================

STYLE_INSTRUCTIONS = {
    "subtle": (
        "Use the attached reference image as a subtle style guide. Borrow only the broad "
        "color palette, surface texture, lighting mood, and visual rhythm. "
        "Do not copy the depicted subject."
    ),
    "balanced": (
        "Use the attached reference image as a style guide. Transfer its visual language, "
        "color palette, material texture, brushwork, lighting, composition rhythm, and "
        "overall mood. Do not copy the depicted subject unless the target prompt asks for it."
    ),
    "strong": (
        "Use the attached reference image as a strong style guide. Strongly adopt its "
        "visual language, color palette, material texture, brushwork, lighting, composition "
        "rhythm, and overall mood, while keeping the target prompt as the subject and scene."
    ),
}


# ============================================================================
# BSAI Krea 2 Style Reference
# ============================================================================

class BSAI_Krea2StyleReference:
    """Krea 2 风格参考节点：将参考图编码为风格 conditioning，可串联多个。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Style reference strength. 0.0 = no influence, 1.0 = standard, 2.0 = very strong.",
                }),
                "style_level": (["subtle", "balanced", "strong"], {
                    "default": "balanced",
                    "tooltip": "Semantic style strength level, controlling how strongly the reference image influences the prompt.",
                }),
            },
            "optional": {
                "vision_resolution": ("INT", {
                    "default": 384,
                    "min": 128,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Resolution to scale the reference image to before the vision encoder.",
                }),
                "style_reference_in": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("style_reference",)
    FUNCTION = "encode"
    CATEGORY = "BSAI/Krea2"

    def encode(self, clip, image, strength, style_level, vision_resolution=384, style_reference_in=None):
        image_for_clip = _resize_image_for_vision(image, vision_resolution)
        instruction = STYLE_INSTRUCTIONS.get(style_level, STYLE_INSTRUCTIONS["balanced"])
        text = f"{instruction}\n\n[Style Reference Image]"

        tokens = clip.tokenize(text, images=[image_for_clip])
        cond = clip.encode_from_tokens_scheduled(tokens)

        if strength != 1.0:
            cond = [
                [c * strength, extra] if isinstance(c, torch.Tensor) else [c, extra]
                for c, extra in cond
            ]

        if style_reference_in is not None:
            merged = list(style_reference_in) + list(cond)
            return (merged,)
        return (cond,)


# ============================================================================
# BSAI Krea 2 Image — 一体化生图节点
# ============================================================================

class BSAI_Krea2Image:
    """BSAI Krea 2 一体化生图节点（Krea 3 语义对齐：最多 8 张风格参考图）。

    接收：模型、CLIP、VAE、正面/负面提示词、最多 8 个可选风格参考图（含各自 strength/level）、
          以及外部 style_reference（支持 CONDITIONING 或 KREA_STYLE_REF）。
    输出：最终图像。
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "style_reference": ("CONDITIONING", {
                "tooltip": "CONDITIONING style condition from the BSAI Krea 2 Style Reference node.",
            }),
            "krea_style_reference": ("KREA_STYLE_REF", {
                "tooltip": "KREA_STYLE_REF output from the official Krea 2 Style Reference node (URL images are downloaded and encoded locally).",
            }),
        }
        for i in range(1, KREA_MAX_STYLE_IMAGES + 1):
            optional[f"style_image_{i}"] = ("IMAGE", {
                "tooltip": f"Style reference image {i} of {KREA_MAX_STYLE_IMAGES} (Krea 3 style: up to 8 reference images).",
            })
            optional[f"style_strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": f"Strength of style reference image {i}.",
            })
            optional[f"style_level_{i}"] = (["subtle", "balanced", "strong"], {
                "default": "balanced",
                "tooltip": f"Semantic style level of reference image {i}.",
            })
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamic_prompts": True,
                    "tooltip": "Positive prompt describing the image content you want to generate.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamic_prompts": True,
                    "tooltip": "Negative prompt describing what you do not want in the image.",
                }),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1344, "min": 256, "max": 2048, "step": 32}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/Krea2"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "BSAI Krea 2 all-in-one image generation node, aligned with the official Krea 3 "
        "semantics: up to 8 style reference images, each with its own strength and level. "
        "Also accepts CONDITIONING style references (BSAI Krea 2 Style Reference) and "
        "official KREA_STYLE_REF chains (URL images are downloaded and encoded locally)."
    )

    def generate(
        self,
        model, clip, vae,
        positive_prompt, negative_prompt,
        width, height,
        steps, cfg,
        sampler, scheduler,
        seed, denoise,
        style_image_1=None, style_strength_1=1.0, style_level_1="balanced",
        style_image_2=None, style_strength_2=1.0, style_level_2="balanced",
        style_image_3=None, style_strength_3=1.0, style_level_3="balanced",
        style_image_4=None, style_strength_4=1.0, style_level_4="balanced",
        style_image_5=None, style_strength_5=1.0, style_level_5="balanced",
        style_image_6=None, style_strength_6=1.0, style_level_6="balanced",
        style_image_7=None, style_strength_7=1.0, style_level_7="balanced",
        style_image_8=None, style_strength_8=1.0, style_level_8="balanced",
        style_reference=None,
        krea_style_reference=None,
    ):
        # --- 收集所有风格条件 ---
        style_conds = []

        # 1) 处理 BSAI style_reference 输入 (CONDITIONING)
        if style_reference is not None:
            style_conds.extend(list(style_reference))

        # 2) Process the official Krea style_reference input (KREA_STYLE_REF)
        if krea_style_reference is not None:
            if _is_krea_style_ref(krea_style_reference):
                logger.info(f"[BSAI Krea2] Detected official Krea Style Reference ({len(krea_style_reference)} images); downloading and encoding locally...")
                krea_cond = _krea_style_ref_to_conditioning(clip, krea_style_reference)
                style_conds.extend(krea_cond)
            else:
                logger.warning("[BSAI Krea2] krea_style_reference has an unknown format; ignoring.")

        # 3) Process embedded style images (up to 8, Krea 3 semantics)
        ref_pairs = []
        for i in range(1, KREA_MAX_STYLE_IMAGES + 1):
            img = locals().get(f"style_image_{i}")
            strength = locals().get(f"style_strength_{i}", 1.0) or 1.0
            level = locals().get(f"style_level_{i}", "balanced") or "balanced"
            if img is not None:
                ref_pairs.append((img, strength, level))
        for ref_img, ref_strength, ref_level in ref_pairs:
            if ref_img is not None:
                resized = _resize_image_for_vision(ref_img, 384)
                instruction = STYLE_INSTRUCTIONS.get(ref_level, STYLE_INSTRUCTIONS["balanced"])
                text = f"{instruction}\n\n[Style Reference Image]"
                tokens = clip.tokenize(text, images=[resized])
                cond = clip.encode_from_tokens_scheduled(tokens)
                if ref_strength != 1.0:
                    cond = [
                        [c * ref_strength, extra] if isinstance(c, torch.Tensor) else [c, extra]
                        for c, extra in cond
                    ]
                style_conds.extend(cond)

        # --- 编码正面/负面提示词 ---
        positive_tokens = clip.tokenize(positive_prompt)
        positive_cond = clip.encode_from_tokens_scheduled(positive_tokens)

        negative_tokens = clip.tokenize(negative_prompt)
        negative_cond = clip.encode_from_tokens_scheduled(negative_tokens)

        # --- 合并风格条件到正面 conditioning ---
        if style_conds:
            positive_cond = list(positive_cond) + style_conds

        # --- 创建空 Latent ---
        latent_channels = 4
        inner = getattr(model, "model", None)
        if inner is not None:
            lf = getattr(inner, "latent_format", None)
            if lf is not None:
                latent_channels = getattr(lf, "latent_channels", 4)
        elif hasattr(model, "latent_channels"):
            latent_channels = model.latent_channels

        latent = torch.zeros(
            [1, latent_channels, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )

        # 修复 latent channels 和维度（如 Krea2 使用 Wan21: latent_dimensions=3, 16ch）
        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        # --- 采样 ---
        sampler = comfy.samplers.KSampler(
            model=model,
            steps=steps,
            device=comfy.model_management.get_torch_device(),
            sampler=sampler,
            scheduler=scheduler,
            denoise=denoise,
        )
        sigmas = sampler.calculate_sigmas(steps)
        noise = comfy.sample.prepare_noise(latent, seed)

        samples = sampler.sample(
            noise=noise,
            sigmas=sigmas,
            cfg=cfg,
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent,
            seed=seed,
        )

        # --- VAE 解码 ---
        # vae.decode 内部做 movedim(1,-1) 返回 [B, H, W, C] 或 [B, T, H, W, C]（5D）。
        # 对于 5D 输出（视频 VAE），合并 batch*时间维为标准 IMAGE 格式。
        decoded = vae.decode(samples)
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        image = decoded.cpu()

        return (image,)


NODE_CLASS_MAPPINGS = {
    "BSAI_Krea2StyleReference": BSAI_Krea2StyleReference,
    "BSAI_Krea2Image": BSAI_Krea2Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_Krea2StyleReference": "BSAI Krea 2 Style Reference",
    "BSAI_Krea2Image": "BSAI Krea 2 Image",
}

WEB_DIRECTORY = "./web"
