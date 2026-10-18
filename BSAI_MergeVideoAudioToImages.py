# -*- coding: utf-8 -*-
"""
BSAI Merge Video+Audio to Images Node for ComfyUI
功能：接收视频路径、原生音频数据、自定义帧率，合并后输出图像序列
作者：BSAI Custom Node
版本：1.3
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
from typing import List
from folder_paths import get_output_directory, get_temp_directory

def get_ffmpeg_path():
    """Return the bundled imageio-ffmpeg binary (no user-supplied executable).

    Security: the previous STRING widget allowed running an arbitrary executable
    path supplied by the user/workflow. We now always use the fixed imageio-ffmpeg
    binary that ships with the ComfyUI Python environment.
    """
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

# 尝试导入必要依赖，缺失时给出明确提示
try:
    import torch
except ImportError:
    torch = None

try:
    import soundfile as sf
except ImportError:
    raise ImportError("soundfile library is missing, please run: pip install soundfile")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow library is missing, please run: pip install pillow")



def _confine_comfy_path(path):
    """Return the realpath of path if it lies inside a ComfyUI base dir, else None."""
    import folder_paths as _fp
    bases = []
    for base in (_fp.get_input_directory(), _fp.get_temp_directory(), _fp.get_output_directory()):
        try:
            bases.append(os.path.realpath(base))
        except Exception:
            pass
    try:
        real = os.path.realpath(str(path or "").strip())
    except Exception:
        return None
    if not real:
        return None
    for base in bases:
        try:
            if os.path.commonpath([base, real]) == base:
                return real
        except Exception:
            continue
    return None

class BSAI_MergeVideoWithAudioToImages:
    """
    BSAI 音视频合并转图像节点
    输入：
        - video_to_path: 视频文件路径（STRING）
        - audio: 原生音频数据（AUDIO）
        - fps: 自定义帧率（FLOAT）
    输出：
        - images: 合并后拆解的图像序列（IMAGE）
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_to_path": ("STRING", {
                    "forceInput": True,
                    "description": "Input video file path (must be inside ComfyUI input/temp/output)"
                }),
                "audio": ("AUDIO", {
                    "forceInput": True,
                    "description": "Raw audio data (ComfyUI AUDIO type)"
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "description": "Frame rate of the output image sequence"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "merge_video_audio_to_images"
    CATEGORY = "BSAI/Video Processing"
    DESCRIPTION = "BSAI - Merge video and audio, extract to image sequence at custom fps"

    def merge_video_audio_to_images(self, video_to_path: str, audio, fps: float) -> tuple[List]:
        # 1. Resolve FFmpeg: always the bundled imageio-ffmpeg binary.
        ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path or not os.path.exists(ffmpeg_path):
            raise RuntimeError("imageio-ffmpeg binary not found in the ComfyUI environment")
        # Path containment: video_to_path must live inside ComfyUI input/temp/output.
        video_to_path = _confine_comfy_path(video_to_path)
        if not video_to_path:
            raise ValueError("video_to_path must be inside ComfyUI input/temp/output directories")
        print(f"[BSAI] Using FFmpeg: {ffmpeg_path}")

        # 2. 创建临时目录（避免文件冲突）
        temp_dir = tempfile.mkdtemp(dir=get_temp_directory(), prefix="bsai_video_audio_")
        merged_video_path = os.path.join(temp_dir, "merged_video.mp4")
        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
        frames_output_pattern = os.path.join(temp_dir, "frame_%08d.png")

        try:
            # Core fix: handle PyTorch Tensor audio input
            if torch is not None and isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()  # Convert to a NumPy array
                print(f"[BSAI] Audio converted from PyTorch Tensor to NumPy array, shape: {audio.shape}")

            # Handle ComfyUI AUDIO data format
            if isinstance(audio, dict):
                if 'waveform' in audio:
                    audio_data = audio['waveform']
                    sample_rate = audio.get('sample_rate', 44100)
                else:
                    raise ValueError("The audio dictionary does not contain 'waveform' data")
            elif isinstance(audio, np.ndarray):
                audio_data = audio
                sample_rate = 44100
            else:
                raise TypeError(f"Unsupported audio input type: {type(audio)}")

            # 3. Normalize different audio data formats
            if len(audio_data.shape) == 1:
                # Mono (samples,) -> (samples, 1)
                audio_data = audio_data[:, np.newaxis]
                print(f"[BSAI] Audio format converted: mono {audio_data.shape} -> (samples, 1)")
            elif len(audio_data.shape) == 2:
                if audio_data.shape[0] < audio_data.shape[1]:
                    # Channels first (channels, samples) -> transpose to (samples, channels)
                    audio_data = audio_data.T
                    print(f"[BSAI] Audio transposed: {audio_data.shape} -> (samples, channels)")
            elif len(audio_data.shape) == 3:
                # Batched shape like [1, channels, samples]
                if audio_data.shape[0] == 1:
                    audio_data = audio_data.squeeze(0)  # Remove batch dimension
                    if audio_data.shape[0] < audio_data.shape[1]:
                        audio_data = audio_data.T
                    print(f"[BSAI] Audio converted from batch {audio_data.shape} to (samples, channels)")
                else:
                    raise ValueError(f"Unsupported 3D audio format: {audio_data.shape}")
            else:
                raise ValueError(f"Unsupported audio format: {audio_data.shape}; expected (samples,) or (samples, channels)")

            # 4. Save audio as a temporary WAV file
            sf.write(temp_audio_path, audio_data, sample_rate)
            print(f"[BSAI] Temporary audio file saved: {temp_audio_path}")

            # 5. Run FFmpeg to merge audio and video and set the frame rate
            merge_cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-i", video_to_path,
                "-i", temp_audio_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-r", str(fps),
                "-shortest",
                "-y",
                merged_video_path
            ]
            subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
            print(f"[BSAI] Video+audio merge complete: {merged_video_path}")

            # 6. Extract the merged video into an image sequence
            extract_cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-i", merged_video_path,
                "-vf", f"fps={fps}",
                "-q:v", "1",
                "-y",
                frames_output_pattern
            ]
            subprocess.run(extract_cmd, check=True, capture_output=True, text=True)

            # 7. Read the images and convert them to ComfyUI IMAGE format
            image_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".png")])
            if not image_files:
                raise RuntimeError("No image frames were extracted; please check that the video file is valid")

            images = []
            for img_file in image_files:
                img_path = os.path.join(temp_dir, img_file)
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img_np = np.array(img).astype(np.float32) / 255.0
                    images.append(img_np)

            # Convert to ComfyUI batch image format (count, height, width, channels)
            images_batch = np.stack(images)
            print(f"[BSAI] Extracted {len(images_batch)} frames, shape: {images_batch.shape}")

            return (images_batch,)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg execution failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Node processing failed: {str(e)}")
        finally:
            # 可选：注释掉下面一行可保留临时文件用于调试
            shutil.rmtree(temp_dir, ignore_errors=True)

# 注册节点到ComfyUI
NODE_CLASS_MAPPINGS = {
    "BSAI_MergeVideoWithAudioToImages": BSAI_MergeVideoWithAudioToImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MergeVideoWithAudioToImages": "BSAI - Merge Video+Audio to Images (Custom FPS)"
}

# 模块导出
if __name__ == "__main__":
    print("BSAI Merge Video+Audio to Images Node loaded")