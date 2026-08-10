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
    """查找FFmpeg路径"""
    # 优先检查系统PATH中的ffmpeg
    for path in os.environ.get('PATH', '').split(os.pathsep):
        ffmpeg_path = os.path.join(path, 'ffmpeg.exe') if os.name == 'nt' else os.path.join(path, 'ffmpeg')
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
    # 检查常见的安装位置
    common_paths = [
        'C:\\ffmpeg\\bin\\ffmpeg.exe',
        'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
    ]
    for ffmpeg_path in common_paths:
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
    return None

# 尝试导入必要依赖，缺失时给出明确提示
try:
    import torch
except ImportError:
    torch = None

try:
    import soundfile as sf
except ImportError:
    raise ImportError("缺少 soundfile 库，请执行：pip install soundfile")

try:
    from PIL import Image
except ImportError:
    raise ImportError("缺少 Pillow 库，请执行：pip install pillow")

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
                    "description": "输入视频文件的路径"
                }),
                "audio": ("AUDIO", {
                    "forceInput": True,
                    "description": "原生音频数据（ComfyUI AUDIO 类型）"
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "description": "输出图像序列的帧率"
                }),
            },
            "optional": {
                "ffmpeg_path": ("STRING", {
                    "default": "",
                    "description": "FFmpeg可执行文件路径（留空则使用ComfyUI自带的FFmpeg）"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "merge_video_audio_to_images"
    CATEGORY = "BSAI/Video Processing"
    DESCRIPTION = "BSAI - 合并视频和音频，按自定义帧率拆解为图像序列"

    def merge_video_audio_to_images(self, video_to_path: str, audio, fps: float, ffmpeg_path: str = "") -> tuple[List]:
        # 1. 确定FFmpeg路径
        if not ffmpeg_path:
            ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path:
            raise RuntimeError("未找到FFmpeg，请在节点中指定FFmpeg路径或确保ComfyUI已正确安装FFmpeg")
        if not os.path.exists(ffmpeg_path):
            raise FileNotFoundError(f"指定的FFmpeg路径不存在：{ffmpeg_path}")
        print(f"[BSAI] 使用FFmpeg：{ffmpeg_path}")

        # 2. 创建临时目录（避免文件冲突）
        temp_dir = tempfile.mkdtemp(dir=get_temp_directory(), prefix="bsai_video_audio_")
        merged_video_path = os.path.join(temp_dir, "merged_video.mp4")
        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
        frames_output_pattern = os.path.join(temp_dir, "frame_%08d.png")

        try:
            # 核心修复：处理PyTorch Tensor格式的音频输入
            if torch is not None and isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()  # 转换为NumPy数组
                print(f"[BSAI] 音频从PyTorch Tensor转换为NumPy数组，形状：{audio.shape}")

            # 处理ComfyUI的AUDIO数据格式
            if isinstance(audio, dict):
                if 'waveform' in audio:
                    audio_data = audio['waveform']
                    sample_rate = audio.get('sample_rate', 44100)
                else:
                    raise ValueError("输入的音频字典中不包含 'waveform' 数据")
            elif isinstance(audio, np.ndarray):
                audio_data = audio
                sample_rate = 44100
            else:
                raise TypeError(f"不支持的音频输入类型：{type(audio)}")

            # 3. 兼容处理不同格式的音频数据
            if len(audio_data.shape) == 1:
                # 单声道 (samples,) → 转换为 (samples, 1)
                audio_data = audio_data[:, np.newaxis]
                print(f"[BSAI] 音频格式转换：单声道 {audio_data.shape} → (samples, 1)")
            elif len(audio_data.shape) == 2:
                if audio_data.shape[0] < audio_data.shape[1]:
                    # 通道在前 (channels, samples) → 转置为 (samples, channels)
                    audio_data = audio_data.T
                    print(f"[BSAI] 音频格式转置：{audio_data.shape} → (samples, channels)")
            elif len(audio_data.shape) == 3:
                # 处理形如 [1, channels, samples] 的批次格式
                if audio_data.shape[0] == 1:
                    audio_data = audio_data.squeeze(0)  # 移除批次维度
                    if audio_data.shape[0] < audio_data.shape[1]:
                        audio_data = audio_data.T
                    print(f"[BSAI] 音频格式从批次 {audio_data.shape} 转换为 (samples, channels)")
                else:
                    raise ValueError(f"不支持的3D音频格式：{audio_data.shape}")
            else:
                raise ValueError(f"不支持的音频格式：{audio_data.shape}，请使用 (samples,) 或 (samples, channels) 格式")

            # 4. 保存音频为临时WAV文件
            sf.write(temp_audio_path, audio_data, sample_rate)
            print(f"[BSAI] 临时音频文件已保存：{temp_audio_path}")

            # 5. 调用FFmpeg合并音视频并设置帧率
            merge_cmd = [
                ffmpeg_path,
                "-hide_banner",  # 隐藏冗余输出
                "-loglevel", "error",  # 仅输出错误信息
                "-i", video_to_path,
                "-i", temp_audio_path,
                "-c:v", "libx264",  # H.264视频编码
                "-preset", "fast",  # 快速编码
                "-crf", "18",  # 高质量视频（0-51，18接近无损）
                "-c:a", "aac",  # AAC音频编码
                "-b:a", "192k",  # 音频码率
                "-r", str(fps),  # 设置输出帧率
                "-shortest",  # 以较短的音视频时长为准
                "-y",  # 覆盖已有文件
                merged_video_path
            ]
            subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
            print(f"[BSAI] 音视频合并完成：{merged_video_path}")

            # 6. 拆解视频为图像序列
            extract_cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-i", merged_video_path,
                "-vf", f"fps={fps}",  # 强制按指定帧率提取帧
                "-q:v", "1",  # 最高图像质量（1-31，1最优）
                "-y",
                frames_output_pattern
            ]
            subprocess.run(extract_cmd, check=True, capture_output=True, text=True)

            # 7. 读取图像并转换为ComfyUI标准IMAGE格式
            image_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".png")])
            if not image_files:
                raise RuntimeError("未提取到任何图像帧，请检查视频文件是否有效")

            images = []
            for img_file in image_files:
                img_path = os.path.join(temp_dir, img_file)
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # 统一转为RGB格式
                    img_np = np.array(img).astype(np.float32) / 255.0  # 归一化到0-1
                    images.append(img_np)

            # 转换为ComfyUI批量图像格式 (数量, 高度, 宽度, 通道)
            images_batch = np.stack(images)
            print(f"[BSAI] 成功提取 {len(images_batch)} 帧图像，形状：{images_batch.shape}")

            return (images_batch,)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg执行失败：{e.stderr}")
        except Exception as e:
            raise RuntimeError(f"节点处理失败：{str(e)}")
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
    print("BSAI Merge Video+Audio to Images Node 已加载")