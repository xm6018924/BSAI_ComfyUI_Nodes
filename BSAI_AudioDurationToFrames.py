"""
BSAI AudioDurationToFrames Node
将音频时长转换为帧数
"""

import torch
import numpy as np

class BSAI_AudioDurationToFrames:
    """
    将音频时长（秒）转换为视频帧数
    
    用途：
    - 根据音频时长计算需要的视频帧数
    - 用于音频驱动的视频生成
    - 同步音频和视频的时长
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "音频时长（秒）"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 1.0,
                    "tooltip": "视频帧率（每秒帧数）"
                }),
            },
            "optional": {
                "round_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否对帧数进行四舍五入"
                }),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("frames", "exact_frames")
    FUNCTION = "calculate_frames"
    CATEGORY = "BSAI/Audio"
    OUTPUT_NODE = False

    def calculate_frames(self, audio_duration, fps, round_frames=True):
        """
        计算音频对应的帧数
        
        Args:
            audio_duration: 音频时长（秒）
            fps: 视频帧率
            round_frames: 是否四舍五入
            
        Returns:
            (frames, exact_frames): (整数帧数, 精确帧数)
        """
        exact_frames = audio_duration * fps
        
        if round_frames:
            frames = int(round(exact_frames))
        else:
            frames = int(exact_frames)
        
        print(f"[BSAI AudioDurationToFrames] 音频时长: {audio_duration}s, FPS: {fps}, 帧数: {frames}")
        
        return (frames, exact_frames)


class BSAI_AudioFramesToDuration:
    """
    将视频帧数转换为音频时长
    
    用途：
    - 根据视频帧数计算对应的音频时长
    - 用于视频驱动的音频生成
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("INT", {
                    "default": 120,
                    "min": 1,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "视频帧数"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 1.0,
                    "tooltip": "视频帧率（每秒帧数）"
                }),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("duration",)
    FUNCTION = "calculate_duration"
    CATEGORY = "BSAI/Audio"
    OUTPUT_NODE = False

    def calculate_duration(self, frames, fps):
        """
        计算帧数对应的音频时长
        
        Args:
            frames: 视频帧数
            fps: 视频帧率
            
        Returns:
            duration: 音频时长（秒）
        """
        duration = frames / fps
        
        print(f"[BSAI AudioFramesToDuration] 帧数: {frames}, FPS: {fps}, 音频时长: {duration:.2f}s")
        
        return (duration,)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "BSAI_AudioDurationToFrames": BSAI_AudioDurationToFrames,
    "BSAI_AudioFramesToDuration": BSAI_AudioFramesToDuration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_AudioDurationToFrames": "BSAI Audio Duration To Frames",
    "BSAI_AudioFramesToDuration": "BSAI Audio Frames To Duration",
}
