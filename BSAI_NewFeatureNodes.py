# -*- coding: utf-8 -*-
"""
BSAI 新特性节点组 —— 对齐 ComfyUI 0.41 官方新能力
====================================================
本模块把官方 0.41 新增/强调的三项能力以 BSAI 经典节点形式落地，
便于直接接入 BSAI 多硬件（GPU/XPU/NPU）工作流：

  1. BSAI_ImageSDRToHDR        —— SDR 转 HDR（Rec.2020 HLG，Lottes 展开，
                                  与官方 ImageSDRToHDR 同算法，自包含实现；
                                  按帧处理视频批，保留 alpha）
  2. BSAI_ReferenceMultiLatent —— 多个 latent 参考（最多 8 张参考图，经 VAE 编码后
                                  注入 positive/negative conditioning 的 reference_latents，
                                  与官方 ReferenceMultiLatent 同语义，经典节点签名）
  3. BSAI_CameraInfoToPrompt   —— 消费官方 CreateCameraInfo / CameraAngle 的
                                  LOAD3D_CAMERA 输出，翻译为自然语言镜头描述提示词
                                  （供 H3 / 视频工作流注入画面提示词）

所有实现均为自包含/轻依赖：不依赖 comfy_extras 私有函数，前端导入无副作用。
"""
import json
import math

try:
    import torch
    import comfy.model_management
except Exception:  # 前端导入时缺依赖不致命，执行时才需要
    torch = None
    comfy_model_management = None


# ============================================================================
# 1) SDR → HDR（Rec.709 sRGB → Rec.2020 HLG，Lottes 展开）
#    常量与算法与 ComfyUI 0.41 官方 ImageSDRToHDR 完全一致（自包含复制）。
# ============================================================================

_REC709_TO_REC2020 = (
    (0.6274038959346991, 0.3292830383778837, 0.0433130656874172),
    (0.0690972893582320, 0.9195403950754587, 0.0113623155663092),
    (0.0163914388751503, 0.0880133078772259, 0.8955952532476238),
)
_REC709_LUMA = (0.2126390058715103, 0.7151686787677559, 0.0721923153607337)
_REC2020_LUMA = (0.2627, 0.6780, 0.0593)
_SDR_WHITE_NITS = 203.0
_HLG_PEAK_NITS = 1000.0
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991072928  # = 0.5 - a*ln(4*a)

_LOTTES_HDR_PARAMETERS = (
    -0.1517456472, -1.465687394, -0.2663205862, -1.11367178,
    -2.938027859, -2.948823929, -2.97376132, -2.978893757,
    -2.991618395, -2.956256151, -2.810376167, -2.666443586,
    -2.584222317, -2.532898188, -2.458465099, -2.309132099,
    -2.216364861, -2.167824268, -2.096168995, -2.062468767,
    -2.130614519, 0.5,
)


def _convert_rgb_primaries(rgb, matrix):
    r, g, b = rgb.unbind(dim=-1)
    return torch.stack([r * row[0] + g * row[1] + b * row[2] for row in matrix], dim=-1)


def _rgb_luminance(rgb, weights):
    return (rgb * rgb.new_tensor(weights)).sum(dim=-1, keepdim=True)


def _sdr_to_hdr_lottes(rgb, target_luminance):
    p = rgb.new_tensor(_LOTTES_HDR_PARAMETERS)
    y = _rgb_luminance(rgb, _REC709_LUMA)
    safe_y = y.clamp_min(1e-6)
    a = 0.7 + 0.6 * p[0].sigmoid()
    d = 0.5 + 2 * p[1].sigmoid()
    peak = 1 + 7 * p[2].sigmoid()
    c = 0.5 + 2 * p[3].sigmoid()
    expanded = safe_y.pow(a) / ((1 / peak - c) * safe_y.pow(a * d) + c).clamp_min(1e-5)
    rational_gain = (expanded / safe_y).clamp_min(1e-3).log2().clamp(-1, 3.5)
    knots = torch.nn.functional.softplus(p[4:21]).cumsum(0)
    x = y.clamp(0, 1).sqrt() * 16
    index = x.long().clamp_max(15)
    curve_gain = knots[index] + (x - index) * (knots[index + 1] - knots[index])
    blend = p[21].clamp(0, 1)
    gain = blend * rational_gain + (1 - blend) * curve_gain
    hdr = _convert_rgb_primaries(rgb, _REC709_TO_REC2020) * gain.exp2()

    if target_luminance is not None:
        white_gain = blend * peak.log2().clamp(-1, 3.5) + (1 - blend) * knots[-1]
        source_white = white_gain.exp2()
        target_white = target_luminance / _SDR_WHITE_NITS
        y_hdr = _rgb_luminance(hdr, _REC2020_LUMA)
        hdr = hdr / (1 + (1 / target_white - 1 / source_white) * y_hdr).clamp_min(1e-6)

    # Inverse HLG OOTF followed by OETF, with zero display black.
    peak_nits = max(_HLG_PEAK_NITS, target_luminance or 0.0)
    gamma = 1.2 + 0.42 * math.log10(peak_nits / 1000.0)
    display = hdr.clamp_min(0) * (_SDR_WHITE_NITS / peak_nits)
    luminance = _rgb_luminance(display, _REC2020_LUMA)
    scene = (display * luminance.clamp_min(1e-12).pow(1 / gamma - 1)).clamp(0, 1)
    low = (3 * scene).sqrt()
    high = _HLG_A * (12 * scene - _HLG_B).clamp_min(1e-12).log() + _HLG_C
    return torch.where(scene <= 1 / 12, low, high)


class BSAI_ImageSDRToHDR:
    """SDR 图像转 Rec.2020 HLG HDR（Lottes 展开，与官方 ImageSDRToHDR 同算法）。

    支持视频帧批（B,H,W,C 4D）逐帧处理；alpha 通道原样保留；
    目标亮度 0 = 使用校准渲染（不归一化暗部）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "SDR image(s) in sRGB. Batch (video frames) supported."}),
                "target_luminance": ("FLOAT", {
                    "default": 650.0, "min": 0.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Peak-white target in nits. 0 keeps the calibrated rendering (dark images are not normalized to this peak).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "BSAI/Color"
    DESCRIPTION = (
        "Convert SDR (sRGB) images/frames to Rec.2020 HLG HDR (Lottes rational expansion, "
        "same algorithm as the official ImageSDRToHDR). Batch-friendly for video frames; "
        "alpha is preserved. Save the result with the HDR color space selected."
    )

    def convert(self, image, target_luminance=650.0):
        if image.shape[-1] == 1:
            image = image.expand(*image.shape[:-1], 3)
        device = comfy.model_management.get_torch_device()
        output = torch.empty_like(image, device=comfy.model_management.intermediate_device(),
                                  dtype=comfy.model_management.intermediate_dtype())
        tgt = float(target_luminance) or None
        for i, frame in enumerate(image):
            rgb = frame[..., :3].to(device=device, dtype=torch.float32)
            output[i, ..., :3] = _sdr_to_hdr_lottes(rgb, tgt).cpu()
        if image.shape[-1] == 4:
            output[..., 3:] = image[..., 3:]
        return (output,)


# ============================================================================
# 2) 多个 latent 参考（与官方 ReferenceMultiLatent 同语义）
# ============================================================================

class BSAI_ReferenceMultiLatent:
    """多参考图 latent 注入节点。

    将最多 8 张参考图经 VAE 编码为 reference_latents，追加到 positive（及可选的
    negative）conditioning 中，供 FLUX3 / Qwen-Image 2.1 等编辑模型使用；
    输出 empty_latent（首张参考 latent×0，或按 megapixels 生成的空 latent）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "megapixels": ("FLOAT", {
                "default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01,
                "tooltip": "Target size of each reference image (preserving aspect ratio). 1.0 ≈ 1024x1024. Used only when resize=true.",
            }),
            "resize": ("BOOLEAN", {
                "default": False,
                "tooltip": "Resize each reference image to the megapixels target before VAE encoding.",
            }),
        }
        for i in range(1, 9):
            optional[f"image_{i}"] = ("IMAGE", {
                "tooltip": f"Reference image {i} of 8 (encoded to latent and appended to conditioning).",
            })
        return {
            "required": {
                "positive": ("CONDITIONING",),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                **optional,
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "empty_latent")
    FUNCTION = "apply"
    CATEGORY = "BSAI/Conditioning"
    DESCRIPTION = (
        "Encode up to 8 reference images into latents and append them to the positive "
        "(and optional negative) conditioning as reference_latents, for edit models "
        "(FLUX3 / Qwen-Image 2.1 etc.). Mirrors the official ReferenceMultiLatent "
        "semantics in a classic-node signature. empty_latent = first reference latent * 0 "
        "or a square empty latent at the selected resolution."
    )

    def apply(self, positive, negative=None, vae=None, megapixels=1.0, resize=False, **kwargs):
        import node_helpers
        from nodes import EmptyLatentImage

        ref_latents = []
        if vae is not None:
            import comfy.utils
            for i in range(1, 9):
                image = kwargs.get(f"image_{i}")
                if image is None:
                    continue
                if resize:
                    height, width = image.shape[1:3]
                    scale = math.sqrt(float(megapixels) * 1024 * 1024 / (width * height))
                    width = max(1, round(width * scale))
                    height = max(1, round(height * scale))
                    image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
                ref_latents.append(vae.encode(image))

        if ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            if negative is not None:
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
            empty_latent = {"samples": ref_latents[0] * 0.0}
        else:
            size = 1024
            if resize:
                size = round(math.sqrt(float(megapixels) * 1024 * 1024))
            empty_latent = EmptyLatentImage().generate(size, size)[0]
        return (positive, negative if negative is not None else positive, empty_latent)


# ============================================================================
# 3) 相机信息 → 镜头描述提示词（消费官方 LOAD3D_CAMERA）
# ============================================================================

_HORIZONTAL_TERMS = (
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
)
_VERTICAL_TERMS = ((-15, "low-angle shot"), (15, "eye-level shot"), (45, "elevated shot"), (None, "high-angle shot"))
_DISTANCE_TERMS = ((2, "wide shot"), (6, "medium shot"), (None, "close-up"))


def _bucket(value, terms):
    for upper, term in terms:
        if upper is None or value < upper:
            return term
    return terms[-1][1]


def _describe_camera_info(camera_info):
    """把 LOAD3D_CAMERA dict 翻译为自然语言镜头描述。"""
    if not isinstance(camera_info, dict):
        return "camera_info unavailable"
    position = camera_info.get("position") or {}
    target = camera_info.get("target") or {}
    px = position.get("x", 0) or 0
    py = position.get("y", 0) or 0
    pz = position.get("z", 0) or 0
    tx = target.get("x", 0) or 0
    ty = target.get("y", 0) or 0
    tz = target.get("z", 0) or 0
    dx, dy, dz = px - tx, py - ty, pz - tz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance < 1e-6:
        horizontal_angle = 0.0
        vertical_angle = 0.0
    else:
        horizontal_angle = math.degrees(math.atan2(dx, dz)) % 360
        vertical_angle = math.degrees(math.asin(max(-1.0, min(1.0, dy / distance))))
    sector = int(((horizontal_angle % 360) + 22.5) // 45) % len(_HORIZONTAL_TERMS)
    h_term = _HORIZONTAL_TERMS[sector]
    v_term = _bucket(vertical_angle, _VERTICAL_TERMS)
    d_term = _bucket(distance, _DISTANCE_TERMS)
    fov = camera_info.get("fov")
    cam_type = camera_info.get("cameraType", "perspective")
    zoom = camera_info.get("zoom", 1.0)
    parts = [h_term, v_term, d_term, cam_type]
    if fov:
        parts.append(f"fov {float(fov):.0f}°")
    if isinstance(zoom, (int, float)) and zoom != 1.0:
        parts.append(f"zoom {float(zoom):.2f}x")
    return ", ".join(parts)


class BSAI_CameraInfoToPrompt:
    """消费官方 CreateCameraInfo / CameraAngle 的 LOAD3D_CAMERA，输出镜头描述提示词。

    输出一：自然语言镜头描述（front view, eye-level shot, medium shot, perspective, fov 35°），
            可直接拼入 H3 / 视频生成提示词；
    输出二：紧凑 JSON（yaw/pitch/distance/fov/zoom/cameraType），供脚本/其他节点使用。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_info": ("LOAD3D_CAMERA", {
                    "tooltip": "camera_info from the official Create Camera Info / Compose Camera Angle Prompt nodes.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "camera_json")
    FUNCTION = "describe"
    CATEGORY = "BSAI/Camera"
    DESCRIPTION = (
        "Translate an official LOAD3D_CAMERA (Create Camera Info / Camera Angle) into a "
        "natural-language shot description prompt for video/image generation, plus a "
        "compact JSON of the camera parameters."
    )

    def describe(self, camera_info):
        prompt = _describe_camera_info(camera_info)
        try:
            cam_json = json.dumps(camera_info, ensure_ascii=False, sort_keys=True)
        except Exception:
            cam_json = "{}"
        return (prompt, cam_json)


NODE_CLASS_MAPPINGS = {
    "BSAI_ImageSDRToHDR": BSAI_ImageSDRToHDR,
    "BSAI_ReferenceMultiLatent": BSAI_ReferenceMultiLatent,
    "BSAI_CameraInfoToPrompt": BSAI_CameraInfoToPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_ImageSDRToHDR": "BSAI SDR to HDR (Rec.2020 HLG)",
    "BSAI_ReferenceMultiLatent": "BSAI Reference Multi Latent (8 refs)",
    "BSAI_CameraInfoToPrompt": "BSAI Camera Info to Prompt",
}
