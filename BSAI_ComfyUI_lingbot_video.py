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

# 将 lingbot_video 模块目录加入 sys.path（相对路径）
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_LINGBOT_VIDEO_LIB = os.path.join(_PLUGIN_DIR, "lingbot_video")
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

# 本地模型存放根目录（相对路径）
_MODEL_ROOT = os.path.join(_PLUGIN_DIR, "..", "..", "models", "lingbot-vision")
_DENSE_MODEL_LOCAL = os.path.normpath(os.path.join(_MODEL_ROOT, "dense-1.3b"))
_DENSE_HF_REPO = "robbyant/lingbot-video-dense-1.3b"

# 缓存
_pipeline_cache = {}
_device = None
_dtype = None


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
    """检查依赖版本是否满足 lingbot-video 最低要求"""
    missing = []
    try:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version("5.0.0"):
            missing.append(f"transformers>={transformers.__version__} (需要 >=5.0.0，当前 {transformers.__version__})")
    except ImportError:
        missing.append("transformers")
    try:
        import diffusers
        from packaging.version import Version
        if Version(diffusers.__version__) < Version("0.37.0"):
            missing.append(f"diffusers (需要 >=0.37.0，当前 {diffusers.__version__})")
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
    """从 HuggingFace 下载模型到本地（自动尝试镜像站）"""
    from huggingface_hub import snapshot_download
    endpoints = [
        None,  # 默认官方
        "https://hf-mirror.com",  # 国内镜像
    ]
    for endpoint in endpoints:
        try:
            label = endpoint or "HuggingFace官方"
            print(f"[BSAI LingBot-Video] 正在从 {label} 下载模型: {hf_repo}")
            print(f"[BSAI LingBot-Video] 下载到: {model_path}")
            os.makedirs(model_path, exist_ok=True)
            kwargs = dict(
                repo_id=hf_repo,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
            if endpoint:
                kwargs["endpoint"] = endpoint
            snapshot_download(**kwargs)
            print(f"[BSAI LingBot-Video] 模型下载完成: {model_path}")
            return True
        except Exception as e:
            print(f"[BSAI LingBot-Video] {label} 下载失败: {e}")
            if endpoint is None:
                continue  # 尝试镜像
            return False
    return False


def _load_pipeline(model_path: str, hf_repo: str, dtype_str: str = "bf16", vae_dtype_str: str = "fp32"):
    """加载 LingBot-Video pipeline"""
    cache_key = (model_path, dtype_str, vae_dtype_str)
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    # 检查本地模型
    if not os.path.isfile(os.path.join(model_path, "model_index.json")):
        print(f"[BSAI LingBot-Video] 本地模型未找到，正在从 HuggingFace 下载...")
        if not _download_model(model_path, hf_repo):
            raise RuntimeError(
                f"无法加载 LingBot-Video 模型。"
                f"\n请手动下载模型到: {model_path}"
                f"\n或确保网络可以访问: https://huggingface.co/{hf_repo}"
            )

    # 解析 dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    transformer_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    vae_dtype = dtype_map.get(vae_dtype_str, torch.float32)

    device = _get_device()

    # 导入 lingbot_video 自定义组件
    try:
        from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
        from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
        from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
    except ImportError as e:
        raise RuntimeError(f"无法导入 lingbot_video 模块: {e}")

    # 手动加载各组件
    from diffusers import AutoencoderKLWan
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    print(f"[BSAI LingBot-Video] 正在加载模型 ({transformer_dtype})...")

    # 加载 scheduler
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(os.path.join(model_path, "scheduler"))
    print("[BSAI LingBot-Video] Scheduler 已加载")

    # 加载 text_encoder + processor
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(model_path, "text_encoder"),
        torch_dtype=transformer_dtype,
    )
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_path, "processor"))
    print("[BSAI LingBot-Video] TextEncoder + Processor 已加载")

    # 加载 transformer
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=transformer_dtype,
    )
    print("[BSAI LingBot-Video] Transformer 已加载")

    # 加载 VAE
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_path, "vae"),
        torch_dtype=vae_dtype,
    )
    print("[BSAI LingBot-Video] VAE 已加载")

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
    print(f"[BSAI LingBot-Video] 模型加载完成 (设备: {device})")
    return pipe


def _save_video(frames: list, fps: int, output_dir: str, filename_prefix: str) -> str:
    """保存视频帧到文件"""
    import folder_paths
    output_base = folder_paths.get_output_directory()
    os.makedirs(output_base, exist_ok=True)

    try:
        import imageio
        filepath = os.path.join(output_base, f"{filename_prefix}.mp4")
        # frames: list of numpy arrays (H, W, C) in RGB uint8
        if frames:
            imageio.mimwrite(filepath, frames, fps=fps, codec="libx264", quality=8)
        else:
            # 如果是单张图
            return _save_image(frames, output_base, filename_prefix)
        return filepath
    except Exception:
        # 回退到逐帧保存为图片
        return _save_image(frames, output_base, filename_prefix)


def _save_image(frames, output_dir: str, filename_prefix: str) -> str:
    """保存为图片"""
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
        print(f"[BSAI LingBot-Video] 保存图片失败: {e}")
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
    """加载 LingBot-Video Dense 1.3B 模型"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "vae_dtype": (["fp32", "bf16", "fp16"], {"default": "fp32"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "tooltip": "自定义模型本地路径，留空使用默认路径 models/lingbot-vision/dense-1.3b/"}),
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
                f"LingBot-Video 依赖不满足:\n  - " + "\n  - ".join(deps) +
                "\n建议升级 transformers>=5.0.0, diffusers>=0.37.0"
            )

        path = model_path.strip() if model_path.strip() else _DENSE_MODEL_LOCAL
        pipe = _load_pipeline(path, _DENSE_HF_REPO, transformer_dtype, vae_dtype)
        return (pipe,)


# ============================================================
# Node: BSAI LingBot-Video Text-to-Video (T2V)
# ============================================================
class BSAI_LingBot_Video_T2V:
    """LingBot-Video 文生视频"""

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
                    raise RuntimeError("无法从 pipeline 输出中提取视频帧")

            comfy_image = _frames_to_comfy_image(frames)
            filepath = _save_video(frames, fps, None, filename_prefix)
            return (comfy_image, filepath,)

        except Exception as e:
            raise RuntimeError(f"LingBot-Video T2V 推理失败: {e}")


# ============================================================
# Node: BSAI LingBot-Video Text-to-Image (T2I)
# ============================================================
class BSAI_LingBot_Video_T2I:
    """LingBot-Video 文生图"""

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

            # 提取图像
            if hasattr(result, "images") and result.images is not None:
                images = result.images
            elif hasattr(result, "frames") and result.frames is not None:
                images = result.frames[0] if isinstance(result.frames, list) else result.frames
            else:
                raise RuntimeError("无法从 pipeline 输出中提取图像")

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
            raise RuntimeError(f"LingBot-Video T2I 推理失败: {e}")


# ============================================================
# Node: BSAI LingBot-Video Image-to-Video (TI2V)
# ============================================================
class BSAI_LingBot_Video_TI2V:
    """LingBot-Video 图文生视频"""

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
            raise RuntimeError(f"输入图像转换失败: {e}")

        # 获取图像尺寸
        w, h = pil_image.size

        try:
            # Pipeline 内部已有精确的 autocast 管理
            torch.cuda.empty_cache()
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

            # 提取视频帧
            if hasattr(result, "frames") and result.frames is not None:
                frames = result.frames[0] if isinstance(result.frames, list) else result.frames
            elif hasattr(result, "videos") and result.videos is not None:
                frames = result.videos[0] if isinstance(result.videos, list) else result.videos
                if isinstance(frames, torch.Tensor):
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
            else:
                raise RuntimeError("无法从 pipeline 输出中提取视频帧")

            comfy_image = _frames_to_comfy_image(frames)
            filepath = _save_video(frames, fps, None, filename_prefix)
            return (comfy_image, filepath,)

        except Exception as e:
            raise RuntimeError(f"LingBot-Video TI2V 推理失败: {e}")


# ============================================================
# Node: BSAI LingBot-Video Unload (卸载模型释放显存)
# ============================================================
class BSAI_LingBot_Video_Unload:
    """卸载 LingBot-Video 模型释放显存"""

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

        print("[BSAI LingBot-Video] 模型已卸载，显存已释放")
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
