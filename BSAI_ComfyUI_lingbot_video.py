"""
BSAI-ComfyUI_lingbot_video.py
LingBot-Video Dense 1.3B ComfyUI 节点
支持: 文生视频(T2V)、文生图(T2I)、图文生视频(TI2V)
模型本地路径: ComfyUI/models/lingbot-vision/dense-1.3b/
自动下载: https://huggingface.co/robbyant/lingbot-video-dense-1.3b
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Optional

# ==== BSAI 插件协同 SDK：加载即自动注册（失败不拖垮插件） ====
try:
    import sys as _bsai_sys, os as _bsai_os
    _BSAI_ORCH_DIR = _bsai_os.path.join(
        _bsai_os.path.dirname(_bsai_os.path.abspath(__file__)),
        "..", "BSAI-ComfyUI-Orchestrator")
    if _bsai_os.path.isdir(_BSAI_ORCH_DIR) and _BSAI_ORCH_DIR not in _bsai_sys.path:
        _bsai_sys.path.insert(0, _BSAI_ORCH_DIR)
    from bsai_orch_client import BSAIOrch  # noqa: E402
    BSAIOrch.register(
        name="BSAI-LingBot-Video",
        kind="video_infer",
        hardware=["cuda"],
    )
except Exception as _bsai_e:  # Registration failure must not break the plugin
    print(f"[BSAI SDK] BSAI-LingBot-Video registration failed (ignored): {_bsai_e}")
# ==== BSAI SDK block end ====


def _bsai_video_lease():
    """BSAI collaboration: video_infer lease (failure does not block inference)."""
    try:
        _a = BSAIOrch.allocate("video_infer", requester="8191")
        if not _a.ok:
            print(f"[BSAI-LingBot-Video] SDK did not obtain video_infer lease ({_a.reason}); continuing with normal inference")
        return _a
    except Exception as _e:
        print(f"[BSAI-LingBot-Video] SDK allocate exception (ignored): {_e}")
        return None


def _bsai_video_release(_a):
    if _a is not None:
        try:
            _a.release()
        except Exception:
            pass


# 将 lingbot_video 模块目录加入 sys.path（相对路径）
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_LINGBOT_VIDEO_LIB = os.path.join(_PLUGIN_DIR, "lingbot_video")
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

# 本地模型存放根目录（相对路径）
_MODEL_ROOT = os.path.normpath(os.path.join(_PLUGIN_DIR, "..", "..", "models", "lingbot-vision"))
_DENSE_MODEL_LOCAL = os.path.normpath(os.path.join(_MODEL_ROOT, "dense-1.3b"))
_DENSE_HF_REPO = "robbyant/lingbot-video-dense-1.3b"

# 缓存
_pipeline_cache = {}
_device = None
_dtype = None


def _confine_model_path(user_path: str) -> str:
    """Resolve a user-supplied model_path and confine it to the allowed model root.

    - Empty/whitespace falls back to the default dense-1.3b directory.
    - Absolute paths and any ".." traversal that escapes _MODEL_ROOT are rejected.
    - Relative paths are resolved against _MODEL_ROOT and verified via realpath+commonpath.
    """
    base = os.path.realpath(_MODEL_ROOT)
    if user_path is None or not str(user_path).strip():
        return _DENSE_MODEL_LOCAL
    raw = str(user_path).strip()
    # Reject explicit traversal and absolute paths outright.
    if os.path.isabs(raw) or ".." in raw.replace("\\", "/").split("/"):
        raise RuntimeError(
            f"Invalid model_path: absolute paths and '..' traversal are not allowed "
            f"(received: {raw!r}). Model path must be relative under {base}."
        )
    candidate = os.path.realpath(os.path.join(base, raw))
    # commonpath raises ValueError on different drives; treat that as escape.
    try:
        if os.path.commonpath([base, candidate]) != base:
            raise ValueError
    except ValueError:
        raise RuntimeError(
            f"Invalid model_path: resolved path escapes the allowed model directory "
            f"(received: {raw!r})."
        )
    return candidate


def _sanitize_filename_prefix(prefix: str) -> str:
    """Return a safe bare filename prefix (no directories, no traversal)."""
    if prefix is None:
        return "LBV"
    name = os.path.basename(str(prefix).replace("\\", "/").strip())
    if not name or name in (".", ".."):
        return "LBV"
    # Strip any residual traversal segments.
    name = name.replace("..", "")
    return name or "LBV"


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _get_dtype():
    global _dtype
    if _dtype is None:
        _dtype = torch.bfloat16 if _get_device() == "cuda" else torch.float32
    return _dtype


def _check_dependencies():
    """Check whether dependency versions meet the minimum lingbot-video requirements."""
    missing = []
    try:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version("5.0.0"):
            missing.append(f"transformers>={transformers.__version__} (requires >=5.0.0, current {transformers.__version__})")
    except ImportError:
        missing.append("transformers")
    try:
        import diffusers
        from packaging.version import Version
        if Version(diffusers.__version__) < Version("0.37.0"):
            missing.append(f"diffusers (requires >=0.37.0, current {diffusers.__version__})")
    except ImportError:
        missing.append("diffusers")
    try:
        import peft
    except ImportError:
        missing.append("peft")
    try:
        import decord
    except ImportError:
        missing.append("decord")
    try:
        import json_repair
    except ImportError:
        missing.append("json_repair")
    return missing


def _download_model(model_path: str, hf_repo: str):
    """Download the model from HuggingFace locally (auto-try mirror endpoints)."""
    from huggingface_hub import snapshot_download
    endpoints = [
        None,  # Default official
        "https://hf-mirror.com",  # Mirror
    ]
    for endpoint in endpoints:
        try:
            label = endpoint or "HuggingFace"
            print(f"[BSAI LingBot-Video] Downloading {hf_repo} from {label}")
            print(f"[BSAI LingBot-Video] Downloading to: {model_path}")
            os.makedirs(model_path, exist_ok=True)
            kwargs = dict(
                repo_id=hf_repo,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
            if endpoint:
                kwargs["endpoint"] = endpoint
            snapshot_download(**kwargs)
            print(f"[BSAI LingBot-Video] Model download complete: {model_path}")
            return True
        except Exception as e:
            print(f"[BSAI LingBot-Video] {label} download failed: {e}")
            if endpoint is None:
                continue  # Try the mirror
            return False
    return False


def _load_pipeline(model_path: str, hf_repo: str, dtype_str: str = "bf16", vae_dtype_str: str = "fp32"):
    """Load the LingBot-Video pipeline."""
    cache_key = (model_path, dtype_str, vae_dtype_str)
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    # Check for a local model
    if not os.path.isfile(os.path.join(model_path, "model_index.json")):
        print(f"[BSAI LingBot-Video] Local model not found, downloading from HuggingFace...")
        if not _download_model(model_path, hf_repo):
            raise RuntimeError(
                f"Unable to load the LingBot-Video model."
                f"\nPlease manually download the model to: {model_path}"
                f"\nor ensure network access to: https://huggingface.co/{hf_repo}"
            )

    # Resolve dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    transformer_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    vae_dtype = dtype_map.get(vae_dtype_str, torch.float32)

    device = _get_device()

    # Import lingbot_video custom components
    try:
        from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
        from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
        from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
    except ImportError as e:
        raise RuntimeError(f"Unable to import the lingbot_video module: {e}")

    # Manually load each component
    from diffusers import AutoencoderKLWan
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    print(f"[BSAI LingBot-Video] Loading model ({transformer_dtype})...")

    # Load scheduler
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(os.path.join(model_path, "scheduler"))
    print("[BSAI LingBot-Video] Scheduler loaded")

    # Load text_encoder + processor
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(model_path, "text_encoder"),
        torch_dtype=transformer_dtype,
    )
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_path, "processor"))
    print("[BSAI LingBot-Video] TextEncoder + Processor loaded")

    # Load transformer
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=transformer_dtype,
    )
    print("[BSAI LingBot-Video] Transformer loaded")

    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_path, "vae"),
        torch_dtype=vae_dtype,
    )
    print("[BSAI LingBot-Video] VAE loaded")

    # 构建 pipeline
    pipe = LingBotVideoPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        scheduler=scheduler,
    )

    pipe = pipe.to(device)

    # 启用 memory savings: 按 text_encoder -> transformer -> vae 顺序
    # 任何时刻只有一个模型在 GPU 上，节省显存避免 OOM
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    # Patch VAE decode: 必须在 enable_model_cpu_offload 之后，
    # 因为 offload 会包装/替换 VAE 模块。
    # 修复: model_cpu_offload 会把 VAE 移到 CPU，decode 前需确保 VAE 在 CUDA，
    # 否则 Conv3d 权重(CPU) 与 latent(CUDA) 不在同一设备，触发 slow_conv3d 错误。
    _original_vae_decode = pipe.vae.decode

    def _patched_vae_decode(z, **kwargs):
        # 确保 VAE 整体移回 CUDA
        pipe.vae.to(z.device if z.device.type == "cuda" else torch.device("cuda"))
        vae_dtype = next(pipe.vae.parameters()).dtype
        if z.device.type == "cuda" and vae_dtype in {torch.bfloat16, torch.float16}:
            z = z.to(vae_dtype)
        with torch.cuda.amp.autocast(enabled=False):
            return _original_vae_decode(z, **kwargs)

    pipe.vae.decode = _patched_vae_decode

    _pipeline_cache[cache_key] = pipe
    print(f"[BSAI LingBot-Video] Model load complete (device: {device})")
    return pipe


def _save_video(frames: list, fps: int, output_dir: str, filename_prefix: str) -> str:
    """Save video frames to a file."""
    import folder_paths
    output_base = folder_paths.get_output_directory()
    os.makedirs(output_base, exist_ok=True)
    filename_prefix = _sanitize_filename_prefix(filename_prefix)

    try:
        import imageio
        filepath = os.path.join(output_base, f"{filename_prefix}.mp4")
        # frames: list of numpy arrays (H, W, C) in RGB uint8
        if frames:
            imageio.mimwrite(filepath, frames, fps=fps, codec="libx264", quality=8)
        else:
            # Single image fallback
            return _save_image(frames, output_base, filename_prefix)
        return filepath
    except Exception:
        # Fall back to per-frame image saving
        return _save_image(frames, output_base, filename_prefix)


def _save_image(frames, output_dir: str, filename_prefix: str) -> str:
    """Save as an image."""
    filename_prefix = _sanitize_filename_prefix(filename_prefix)
    try:
        from PIL import Image
        if frames and len(frames) > 0:
            filepath = os.path.join(output_dir, f"{filename_prefix}.png")
            if isinstance(frames[0], np.ndarray):
                Image.fromarray(frames[0]).save(filepath)
            else:
                frames[0].save(filepath)
            return filepath
    except Exception as e:
        print(f"[BSAI LingBot-Video] Failed to save image: {e}")
    return ""


def _prepare_prompt_json(prompt: str, negative_prompt: str = "") -> dict:
    """将文本 prompt 转换为 lingbot-video 需要的 JSON 格式"""
    prompt_json = {
        "caption": prompt,
    }
    if negative_prompt:
        prompt_json["negative_prompt"] = negative_prompt
    return prompt_json


def _frames_to_comfy_image(frames):
    """将视频帧列表转换为 ComfyUI IMAGE 格式 [B, H, W, C] float32 torch.Tensor"""
    if isinstance(frames, torch.Tensor):
        if frames.ndim == 4 and frames.shape[0] == 1:  # [1, C, H, W] or [1, F, C, H, W]
            frames = frames.squeeze(0)
        if frames.ndim == 4 and frames.shape[0] <= 4:  # [C, F, H, W]
            frames = frames.permute(1, 2, 3, 0)  # [F, H, W, C]
        frames = frames.cpu().numpy()
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)  # [F, H, W, C]
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32) / 255.0
    # 确保 shape 为 [B, H, W, C]
    if frames.ndim == 3:
        frames = frames[None, ...]
    # ComfyUI IMAGE 类型要求 torch.Tensor，不能返回 numpy 数组
    return torch.from_numpy(frames)


# ============================================================
# Node: BSAI LingBot-Video Loader (模型加载器)
# ============================================================
class BSAI_LingBot_Video_Loader:
    """Load the LingBot-Video Dense 1.3B model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "vae_dtype": (["fp32", "bf16", "fp16"], {"default": "fp32"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "tooltip": "Custom local model path; leave empty to use the default models/lingbot-vision/dense-1.3b/"}),
            }
        }

    RETURN_TYPES = ("LBV_PIPE",)
    RETURN_NAMES = ("lingbot_video_pipe",)
    FUNCTION = "load_model"
    CATEGORY = "BSAI/LingBot-Video"
    OUTPUT_NODE = False

    def load_model(self, transformer_dtype: str, vae_dtype: str, model_path: str = ""):
        deps = _check_dependencies()
        if deps:
            raise RuntimeError(
                f"LingBot-Video dependencies not satisfied:\n  - " + "\n  - ".join(deps) +
                "\nRecommended: upgrade transformers>=5.0.0, diffusers>=0.37.0"
            )

        path = _confine_model_path(model_path)
        pipe = _load_pipeline(path, _DENSE_HF_REPO, transformer_dtype, vae_dtype)
        return (pipe,)


# ============================================================
# Node: BSAI LingBot-Video Text-to-Video (T2V)
# ============================================================
class BSAI_LingBot_Video_T2V:
    """LingBot-Video Text-to-Video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lingbot_video_pipe": ("LBV_PIPE",),
                "prompt": ("STRING", {"default": "A cat walking on the grass, high quality", "multiline": True}),
                "width": ("INT", {"default": 832, "min": 128, "max": 1920, "step": 16}),
                "height": ("INT", {"default": 480, "min": 128, "max": 1080, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 300, "step": 1}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 200, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "filename_prefix": ("STRING", {"default": "LBV-T2V"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "output_path",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/LingBot-Video"
    OUTPUT_NODE = True

    def generate(
        self,
        lingbot_video_pipe,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        guidance_scale: float,
        fps: int,
        seed: int,
        negative_prompt: str = "",
        shift: float = 3.0,
        filename_prefix: str = "LBV-T2V",
    ):
        pipe = lingbot_video_pipe
        device = _get_device()
        generator = torch.Generator(device=device).manual_seed(seed)

        try:
            # Pipeline 内部已有精确的 autocast 管理，不需要外层 autocast
            # 外层 autocast 会干扰 VAE decode 的 dtype 控制，并增加显存开销
            torch.cuda.empty_cache()
            _bsai_a = _bsai_video_lease()
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    shift=shift,
                )
            finally:
                _bsai_video_release(_bsai_a)

            # 提取视频帧
            if hasattr(result, "frames") and result.frames is not None:
                frames = result.frames[0] if isinstance(result.frames, list) else result.frames
            elif hasattr(result, "videos") and result.videos is not None:
                frames = result.videos[0] if isinstance(result.videos, list) else result.videos
                if isinstance(frames, torch.Tensor):
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # [F, H, W, C]
            else:
                output = result.images[0] if hasattr(result, "images") else None
                if output is not None:
                    frames = [output]
                else:
                    raise RuntimeError("Unable to extract video frames from the pipeline output")

            comfy_image = _frames_to_comfy_image(frames)
            filepath = _save_video(frames, fps, None, filename_prefix)
            return (comfy_image, filepath,)

        except Exception as e:
            raise RuntimeError(f"LingBot-Video T2V inference failed: {e}")


# ============================================================
# Node: BSAI LingBot-Video Text-to-Image (T2I)
# ============================================================
class BSAI_LingBot_Video_T2I:
    """LingBot-Video Text-to-Image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lingbot_video_pipe": ("LBV_PIPE",),
                "prompt": ("STRING", {"default": "A beautiful sunset over mountains, high quality", "multiline": True}),
                "width": ("INT", {"default": 832, "min": 128, "max": 1920, "step": 16}),
                "height": ("INT", {"default": 480, "min": 128, "max": 1080, "step": 16}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 200, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "filename_prefix": ("STRING", {"default": "LBV-T2I"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "output_path",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/LingBot-Video"
    OUTPUT_NODE = True

    def generate(
        self,
        lingbot_video_pipe,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str = "",
        shift: float = 3.0,
        filename_prefix: str = "LBV-T2I",
    ):
        pipe = lingbot_video_pipe
        device = _get_device()
        generator = torch.Generator(device=device).manual_seed(seed)

        try:
            # Pipeline 内部已有精确的 autocast 管理
            torch.cuda.empty_cache()
            _bsai_a = _bsai_video_lease()
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    height=height,
                    width=width,
                    num_frames=1,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    shift=shift,
                )
            finally:
                _bsai_video_release(_bsai_a)

            # 提取图像
            if hasattr(result, "images") and result.images is not None:
                images = result.images
            elif hasattr(result, "frames") and result.frames is not None:
                images = result.frames[0] if isinstance(result.frames, list) else result.frames
            else:
                raise RuntimeError("Unable to extract image from the pipeline output")

            # 转换为 ComfyUI IMAGE 格式: [B, H, W, C] numpy array
            if isinstance(images, list):
                image = images[0]
            else:
                image = images

            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()

            if isinstance(image, np.ndarray):
                if image.ndim == 3:
                    image = image[None, ...]  # [1, H, W, C]
                if image.dtype != np.float32:
                    image = image.astype(np.float32) / 255.0

            filepath = _save_image(images if isinstance(images, list) else [images], None, filename_prefix)
            # ComfyUI IMAGE 类型要求 torch.Tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            return (image, filepath,)

        except Exception as e:
            raise RuntimeError(f"LingBot-Video T2I inference failed: {e}")


# ============================================================
# Node: BSAI LingBot-Video Image-to-Video (TI2V)
# ============================================================
class BSAI_LingBot_Video_TI2V:
    """LingBot-Video Image+Text-to-Video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lingbot_video_pipe": ("LBV_PIPE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Camera zooms in slowly, cinematic", "multiline": True}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 300, "step": 1}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 200, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "filename_prefix": ("STRING", {"default": "LBV-TI2V"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "output_path",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/LingBot-Video"
    OUTPUT_NODE = True

    def generate(
        self,
        lingbot_video_pipe,
        image,
        prompt: str,
        num_frames: int,
        steps: int,
        guidance_scale: float,
        fps: int,
        seed: int,
        negative_prompt: str = "",
        shift: float = 3.0,
        filename_prefix: str = "LBV-TI2V",
    ):
        pipe = lingbot_video_pipe
        device = _get_device()
        generator = torch.Generator(device=device).manual_seed(seed)

        # 构建 I2V pipeline（复用已加载的组件）
        from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline
        i2v_pipe = LingBotVideoImageToVideoPipeline(
            transformer=pipe.transformer,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            processor=pipe.processor,
            scheduler=pipe.scheduler,
        )

        # 将 ComfyUI IMAGE [B, H, W, C] float32 -> PIL Image
        try:
            from PIL import Image as PILImage
            img_np = image[0].astype(np.uint8)
            pil_image = PILImage.fromarray(img_np).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Input image conversion failed: {e}")

        # 获取图像尺寸
        w, h = pil_image.size

        try:
            # Pipeline 内部已有精确的 autocast 管理
            torch.cuda.empty_cache()
            _bsai_a = _bsai_video_lease()
            try:
                result = i2v_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    image=pil_image,
                    height=h,
                    width=w,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    shift=shift,
                )
            finally:
                _bsai_video_release(_bsai_a)

            # 提取视频帧
            if hasattr(result, "frames") and result.frames is not None:
                frames = result.frames[0] if isinstance(result.frames, list) else result.frames
            elif hasattr(result, "videos") and result.videos is not None:
                frames = result.videos[0] if isinstance(result.videos, list) else result.videos
                if isinstance(frames, torch.Tensor):
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
            else:
                raise RuntimeError("Unable to extract video frames from the pipeline output")

            comfy_image = _frames_to_comfy_image(frames)
            filepath = _save_video(frames, fps, None, filename_prefix)
            return (comfy_image, filepath,)

        except Exception as e:
            raise RuntimeError(f"LingBot-Video TI2V inference failed: {e}")


# ============================================================
# Node: BSAI LingBot-Video Unload (release VRAM)
# ============================================================
class BSAI_LingBot_Video_Unload:
    """Unload the LingBot-Video model and release VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lingbot_video_pipe": ("LBV_PIPE",)}}

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "BSAI/LingBot-Video"
    OUTPUT_NODE = False

    def unload(self, lingbot_video_pipe):
        global _pipeline_cache
        # 清除缓存
        keys_to_remove = [k for k in _pipeline_cache if _pipeline_cache[k] is lingbot_video_pipe]
        for k in keys_to_remove:
            del _pipeline_cache[k]

        # 释放显存
        try:
            if hasattr(lingbot_video_pipe, "to"):
                lingbot_video_pipe.to("cpu")
            del lingbot_video_pipe
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[BSAI LingBot-Video] Model unloaded, VRAM released")
        return ()


# ============================================================
# 注册节点
# ============================================================

# 自定义类型注册（让 ComfyUI 识别 LBV_PIPE 类型）
try:
    # ComfyUI 会自动处理自定义类型，无需额外注册
    pass
except Exception:
    pass

NODE_CLASS_MAPPINGS = {
    "BSAI_LingBot_Video_Loader": BSAI_LingBot_Video_Loader,
    "BSAI_LingBot_Video_T2V": BSAI_LingBot_Video_T2V,
    "BSAI_LingBot_Video_T2I": BSAI_LingBot_Video_T2I,
    "BSAI_LingBot_Video_TI2V": BSAI_LingBot_Video_TI2V,
    "BSAI_LingBot_Video_Unload": BSAI_LingBot_Video_Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_LingBot_Video_Loader": "BSAI LingBot-Video Loader (Dense 1.3B)",
    "BSAI_LingBot_Video_T2V": "BSAI LingBot-Video Text-to-Video",
    "BSAI_LingBot_Video_T2I": "BSAI LingBot-Video Text-to-Image",
    "BSAI_LingBot_Video_TI2V": "BSAI LingBot-Video Image-to-Video",
    "BSAI_LingBot_Video_Unload": "BSAI LingBot-Video Unload",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
