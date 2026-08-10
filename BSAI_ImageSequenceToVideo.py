import os
import subprocess
import tempfile
import torch
from PIL import Image
import numpy as np
from comfy.model_management import get_torch_device
import folder_paths

class BSAI_ImageSequenceToVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images1": ("IMAGE",),
                "images2": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "display": "number",
                }),
                "video_name": ("STRING", {"default": "merged_video"}),
                "output_path": ("STRING", {"default": "C:/Users/Desktop/output"}),
                "device": (["CPU", "GPU"], {"default": "CPU",}),
                "video_format": (["mp4", "avi", "mov", "mkv"], {"default": "mp4",}),
            },
            "optional": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "image_sequence_to_video"
    CATEGORY = "BSAI"

    def image_sequence_to_video(self, images1, images2, fps, video_name, output_path, device, video_format, audio1=None, audio2=None):
        try:
            # 清理路径中的引号和其他非法字符
            output_path = output_path.strip().strip('"').strip("'")
            # 确保输出目录存在
            output_path = os.path.abspath(output_path)
            if not output_path or not os.path.exists(output_path):
                output_path = os.path.join(os.path.expanduser("~"), "Desktop", "output")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # 创建临时目录来存储图像帧和音频文件
            with tempfile.TemporaryDirectory() as temp_dir:
                # 保存所有图像序列为连续编号的文件
                all_images = list(images1) + list(images2)
                for i, img in enumerate(all_images):
                    img_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    self.save_tensor_image(img, img_path)

                # 获取所有图像并按文件名排序
                image_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
                if not image_files:
                    raise ValueError("No images found to create video")

                # 构建输出视频路径
                output_video = os.path.join(output_path, f"{video_name}.{video_format}")

                # 获取图像尺寸
                first_img_path = os.path.join(temp_dir, image_files[0])
                width, height = self.get_image_size(first_img_path)

                # 处理音频（如果有）
                audio_file = None
                if audio1 is not None or audio2 is not None:
                    audio_file = self.save_audio(temp_dir, audio1, audio2)

                # 构建 FFmpeg 命令
                if device == "CPU":
                    cmd = [
                        'ffmpeg',
                        '-framerate', str(fps),
                        '-i', f'{temp_dir}/frame_%06d.png',
                        '-vf', f'scale={width}:{height}',
                        '-c:v', 'libx264',
                        '-crf', '28',
                        '-pix_fmt', 'yuv420p',
                    ]
                else:
                    cmd = [
                        'ffmpeg',
                        '-framerate', str(fps),
                        '-i', f'{temp_dir}/frame_%06d.png',
                        '-vf', f'scale={width}:{height}',
                        '-c:v', 'h264_nvenc',
                        '-preset', 'fast',
                        '-cq', '22',
                        '-pix_fmt', 'yuv420p',
                    ]

                # 如果有音频，添加到命令中
                if audio_file:
                    cmd.extend(['-i', audio_file, '-c:a', 'aac', '-b:a', '192k'])
                
                cmd.extend(['-y', output_video])

                # 执行 FFmpeg 命令
                result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8')
                    print(f"Error: {error_msg}")
                    raise ValueError(f"FFmpeg error: {error_msg}")

                # 输出标准输出信息
                print(result.stdout.decode('utf-8'))

                # 构建保存结果对象
                output_dir = folder_paths.get_output_directory()
                relative_path = os.path.relpath(output_video, output_dir)
                subfolder = os.path.dirname(relative_path)
                filename = os.path.basename(output_video)
                
                # 返回视频路径和 UI 输出，用于预览和下载
                return {
                    "ui": {
                        "images": [{
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": "output"
                        }],
                        "animated": [True]
                    },
                    "result": (output_video,)
                }
        except Exception as e:
            raise ValueError(str(e))

    def save_tensor_image(self, tensor, path):
        """将张量图像保存为 PNG 文件"""
        # 确保张量在 CPU 上
        tensor = tensor.cpu()
        # 转换为 numpy 数组并裁剪到 0-255 范围
        img_array = np.clip(255. * tensor.numpy().squeeze(), 0, 255).astype(np.uint8)
        # 转换为 PIL 图像并保存
        Image.fromarray(img_array).save(path)

    def get_image_size(self, image_path):
        """获取图像尺寸"""
        with Image.open(image_path) as img:
            return img.size

    def save_audio(self, temp_dir, audio1=None, audio2=None):
        """保存音频文件并返回路径"""
        audio_files = []
        
        if audio1 is not None and 'waveform' in audio1:
            audio1_path = os.path.join(temp_dir, "audio1.wav")
            self.save_waveform(audio1['waveform'], audio1_path)
            audio_files.append(audio1_path)
        
        if audio2 is not None and 'waveform' in audio2:
            audio2_path = os.path.join(temp_dir, "audio2.wav")
            self.save_waveform(audio2['waveform'], audio2_path)
            audio_files.append(audio2_path)
        
        if len(audio_files) == 0:
            return None
        elif len(audio_files) == 1:
            return audio_files[0]
        else:
            # 合并两个音频文件
            merged_audio = os.path.join(temp_dir, "merged_audio.wav")
            concat_file = os.path.join(temp_dir, "audio_concat.txt")
            with open(concat_file, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file, '-c', 'copy', '-y', merged_audio]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Audio merge error: {result.stderr.decode('utf-8')}")
                return audio_files[0]  # 如果合并失败，返回第一个音频
            
            return merged_audio

    def save_waveform(self, waveform, path):
        """保存波形为 WAV 文件"""
        import scipy.io.wavfile as wavfile
        
        # 确保波形在 CPU 上并转换为 numpy 数组
        waveform_np = waveform.cpu().numpy().squeeze()
        
        # 归一化到 -1 到 1 范围
        waveform_np = np.clip(waveform_np, -1, 1)
        
        # 转换为 16 位整数
        waveform_int = (waveform_np * 32767).astype(np.int16)
        
        # 保存为 WAV 文件
        wavfile.write(path, 44100, waveform_int)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "BSAI_ImageSequenceToVideo": BSAI_ImageSequenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_ImageSequenceToVideo": "BSAI Image Sequence To Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
