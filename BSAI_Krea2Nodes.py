"""
BSAI Krea 2 风格参考生图节点组

提供 BSAI Krea 2 Style Reference 和 BSAI Krea 2 Image 两个节点：
- BSAI Krea 2 Style Reference: 将参考图通过 Krea2 CLIP vision 编码为风格条件
- BSAI Krea 2 Image: 一体化生图节点（模型+提示词+2个风格参考+采样+VAE解码）

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

# ============================================================================
# Helper: 从 URL 下载图片为 tensor
# ============================================================================

def _download_image_from_url(url: str):
    """从 HTTP URL 下载图片并返回 ComfyUI IMAGE tensor [B, H, W, C] (0~1 float32)。"""
    try:
        import urllib.request
        from PIL import Image
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
        logger.warning(f"[BSAI Krea2] 无法下载风格参考图 URL: {url}，错误: {e}")
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
    "轻微": (
        "Use the attached reference image as a subtle style guide. Borrow only the broad "
        "color palette, surface texture, lighting mood, and visual rhythm. "
        "Do not copy the depicted subject."
    ),
    "平衡": (
        "Use the attached reference image as a style guide. Transfer its visual language, "
        "color palette, material texture, brushwork, lighting, composition rhythm, and "
        "overall mood. Do not copy the depicted subject unless the target prompt asks for it."
    ),
    "强烈": (
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
                    "tooltip": "风格参考强度。0.0 = 无影响，1.0 = 标准强度，2.0 = 极强影响。",
                }),
                "style_level": (["轻微", "平衡", "强烈"], {
                    "default": "平衡",
                    "tooltip": "语义风格强度级别，控制参考图对提示词的影响程度。",
                }),
            },
            "optional": {
                "vision_resolution": ("INT", {
                    "default": 384,
                    "min": 128,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "参考图送入 vision encoder 前的缩放分辨率。",
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
        instruction = STYLE_INSTRUCTIONS.get(style_level, STYLE_INSTRUCTIONS["平衡"])
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
    """BSAI Krea 2 一体化生图节点。

    接收：模型、CLIP、VAE、正面/负面提示词、2个可选风格参考图（含strength）、
          以及外部 style_reference（支持 CONDITIONING 或 KREA_STYLE_REF）。
    输出：最终图像。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "正向提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamic_prompts": True,
                    "tooltip": "正面提示词，描述你想要生成的图像内容。",
                }),
                "负面提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamic_prompts": True,
                    "tooltip": "负面提示词，描述你不想在图像中出现的内容。",
                }),
                "宽度": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "高度": ("INT", {"default": 1344, "min": 256, "max": 2048, "step": 32}),
                "步数": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "风格图1": ("IMAGE", {"tooltip": "第一张风格参考图（如科幻城市风格）。"}),
                "风格强度1": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "第一张风格参考图的强度。",
                }),
                "风格级别1": (["轻微", "平衡", "强烈"], {"default": "平衡"}),
                "风格图2": ("IMAGE", {"tooltip": "第二张风格参考图（如角色风格）。"}),
                "风格强度2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "第二张风格参考图的强度。",
                }),
                "风格级别2": (["轻微", "平衡", "强烈"], {"default": "平衡"}),
                "style_reference": ("CONDITIONING", {
                    "tooltip": "来自 BSAI Krea 2 Style Reference 节点的 CONDITIONING 风格条件。",
                }),
                "krea_style_reference": ("KREA_STYLE_REF", {
                    "tooltip": "来自官方 Krea 2 Style Reference 节点的 KREA_STYLE_REF 输出（自动下载 URL 图片并本地编码）。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/Krea2"
    OUTPUT_NODE = True

    def generate(
        self,
        model, clip, vae,
        正向提示词, 负面提示词,
        宽度, 高度,
        步数, cfg,
        采样器, 调度器,
        种子, denoise,
        风格图1=None, 风格强度1=1.0, 风格级别1="平衡",
        风格图2=None, 风格强度2=1.0, 风格级别2="平衡",
        style_reference=None,
        krea_style_reference=None,
    ):
        # --- 收集所有风格条件 ---
        style_conds = []

        # 1) 处理 BSAI style_reference 输入 (CONDITIONING)
        if style_reference is not None:
            style_conds.extend(list(style_reference))

        # 2) 处理官方 Krea style_reference 输入 (KREA_STYLE_REF)
        if krea_style_reference is not None:
            if _is_krea_style_ref(krea_style_reference):
                logger.info(f"[BSAI Krea2] 检测到官方 Krea Style Reference ({len(krea_style_reference)} 张)，正在下载并本地编码...")
                krea_cond = _krea_style_ref_to_conditioning(clip, krea_style_reference)
                style_conds.extend(krea_cond)
            else:
                logger.warning("[BSAI Krea2] krea_style_reference 格式未知，已忽略。")

        # 3) 处理内嵌风格图
        for ref_img, ref_strength, ref_level in [
            (风格图1, 风格强度1, 风格级别1),
            (风格图2, 风格强度2, 风格级别2),
        ]:
            if ref_img is not None:
                resized = _resize_image_for_vision(ref_img, 384)
                instruction = STYLE_INSTRUCTIONS.get(ref_level, STYLE_INSTRUCTIONS["平衡"])
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
        positive_tokens = clip.tokenize(正向提示词)
        positive_cond = clip.encode_from_tokens_scheduled(positive_tokens)

        negative_tokens = clip.tokenize(负面提示词)
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
            [1, latent_channels, 高度 // 8, 宽度 // 8],
            device=comfy.model_management.intermediate_device(),
        )

        # 修复 latent channels 和维度（如 Krea2 使用 Wan21: latent_dimensions=3, 16ch）
        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        # --- 采样 ---
        sampler = comfy.samplers.KSampler(
            model=model,
            steps=步数,
            device=comfy.model_management.get_torch_device(),
            sampler=采样器,
            scheduler=调度器,
            denoise=denoise,
        )
        sigmas = sampler.calculate_sigmas(步数)
        noise = comfy.sample.prepare_noise(latent, 种子)

        samples = sampler.sample(
            noise=noise,
            sigmas=sigmas,
            cfg=cfg,
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent,
            seed=种子,
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
