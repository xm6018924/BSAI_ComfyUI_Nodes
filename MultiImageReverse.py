from __future__ import annotations
import copy
import random
import re
import time

from ollama import Client
import numpy as np
import base64
from io import BytesIO
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass, field
from pydantic.json_schema import JsonSchemaValue

# For type checking only. Torch is not installed at runtime
if TYPE_CHECKING:
    import torch

# 尝试导入ComfyUI的server模块，如果失败则跳过（用于独立测试）
try:
    from server import PromptServer
    from aiohttp import web
    
    @PromptServer.instance.routes.post("/bsai_ollama/get_models")
    async def get_models_endpoint(request):
        data = await request.json()

        url = data.get("url")
        client = Client(host=url)

        models = client.list().get('models', [])

        try:
            models = [model['model'] for model in models]
            return web.json_response(models)
        except Exception as e:
            models = [model['name'] for model in models]
            return web.json_response(models)
except ImportError:
    # 独立测试时跳过server相关代码
    pass

class BSAI_MultiImageVideoReverse:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "你是一位专业的图像和视频分析专家，擅长从多张图像和视频帧中分析并提取共同特征和主题。请详细分析以下内容，提取关键元素、风格、色彩、构图等信息，并生成详细的描述。",
                    "tooltip": "系统提示词 - 用于设置模型的角色和行为"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "请详细分析这些图像和视频帧，包括所有细节，并生成一个综合描述。",
                    "tooltip": "用户提示词 - 你希望模型执行的任务"
                }),
                "model": (["qwen3:latest", "qwen3.5:latest", "qwen3.6:latest", "qwen3.8:latest", "gemma4:31b", "custom"],
                          {"default": "qwen3.8:latest", "tooltip": "选择一个Ollama模型进行推理"}),
                "url": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "Ollama服务器的URL。默认值指向本地实例，使用ollama的默认端口配置。"
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "自定义模型名称（当选择custom时使用）"
                }),
            },
            "optional": {
                # 图像输入（最多10张）
                "image1": ("IMAGE", {"forceInput": False, "tooltip": "第一张输入图像"}),
                "image2": ("IMAGE", {"forceInput": False, "tooltip": "第二张输入图像"}),
                "image3": ("IMAGE", {"forceInput": False, "tooltip": "第三张输入图像"}),
                "image4": ("IMAGE", {"forceInput": False, "tooltip": "第四张输入图像"}),
                "image5": ("IMAGE", {"forceInput": False, "tooltip": "第五张输入图像"}),
                "image6": ("IMAGE", {"forceInput": False, "tooltip": "第六张输入图像"}),
                "image7": ("IMAGE", {"forceInput": False, "tooltip": "第七张输入图像"}),
                "image8": ("IMAGE", {"forceInput": False, "tooltip": "第八张输入图像"}),
                "image9": ("IMAGE", {"forceInput": False, "tooltip": "第九张输入图像"}),
                "image10": ("IMAGE", {"forceInput": False, "tooltip": "第十张输入图像"}),
                # 视频输入（最多5个视频）
                "video1": ("IMAGE", {"forceInput": False, "tooltip": "第一个视频（帧序列）"}),
                "video2": ("IMAGE", {"forceInput": False, "tooltip": "第二个视频（帧序列）"}),
                "video3": ("IMAGE", {"forceInput": False, "tooltip": "第三个视频（帧序列）"}),
                "video4": ("IMAGE", {"forceInput": False, "tooltip": "第四个视频（帧序列）"}),
                "video5": ("IMAGE", {"forceInput": False, "tooltip": "第五个视频（帧序列）"}),
                # 视频采样参数
                "video_sample_interval": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, 
                                                  "tooltip": "视频帧采样间隔，每隔多少帧取一帧"}),
                "max_frames_per_video": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, 
                                                 "tooltip": "每个视频最多采样的帧数"}),
                # 推理参数
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05, "tooltip": "控制生成文本的创意程度，值越高越创意"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "控制生成文本的多样性，值越高越多样"}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 100000, "step": 1, "tooltip": "生成的最大令牌数"}),
            }
        }

    # 允许用户手动添加更多输入端口
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return str(time.time())

    RETURN_TYPES = ("STRING", "STRING", "*")
    RETURN_NAMES = ("response", "thinking", "optional_prompt_list")
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "multi_media_reverse"
    CATEGORY = "BSAI"
    DESCRIPTION = "BSAI 多图像视频反推节点，使用本地Ollama模型分析多张图像和视频帧并生成描述"

    def _process_image_tensor(self, img_tensor):
        """处理单个图像张量，转换为base64编码"""
        images_b64 = []
        try:
            # 检查是否是torch张量
            if hasattr(img_tensor, 'cpu'):
                print(f"[BSAI_MultiImageVideoReverse] 处理图像，形状: {img_tensor.shape}")
                # 处理批量图像（4D张量：B, H, W, C）
                if len(img_tensor.shape) == 4:
                    for idx in range(img_tensor.shape[0]):
                        img_np = 255. * img_tensor[idx].cpu().numpy()
                        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        images_b64.append(img_bytes)
                        print(f"[BSAI_MultiImageVideoReverse] 成功处理批量图像 {idx+1}/{img_tensor.shape[0]}")
                elif len(img_tensor.shape) == 3:
                    # 单张图像（3D张量：H, W, C）
                    img_np = 255. * img_tensor.cpu().numpy()
                    img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    images_b64.append(img_bytes)
                    print(f"[BSAI_MultiImageVideoReverse] 成功处理单个图像")
                else:
                    print(f"[BSAI_MultiImageVideoReverse] WARNING: 图像张量维度不支持: {img_tensor.shape}")
            else:
                # 如果不是torch张量，尝试直接处理
                print(f"[BSAI_MultiImageVideoReverse] WARNING: 图像不是torch张量，跳过处理")
        except Exception as e:
            error_msg = f"处理图像时出错: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        
        return images_b64

    def _process_video_tensor(self, video_tensor, sample_interval=10, max_frames=5):
        """处理视频张量，采样关键帧转换为base64编码"""
        images_b64 = []
        try:
            if video_tensor is None:
                return images_b64
                
            if hasattr(video_tensor, 'cpu'):
                print(f"[BSAI_MultiImageVideoReverse] 处理视频，形状: {video_tensor.shape}")
                # 视频是4D张量：B, H, W, C（B是帧数）
                if len(video_tensor.shape) == 4:
                    total_frames = video_tensor.shape[0]
                    # 计算采样索引
                    indices = list(range(0, total_frames, sample_interval))[:max_frames]
                    print(f"[BSAI_MultiImageVideoReverse] 视频共 {total_frames} 帧，采样 {len(indices)} 帧: {indices}")
                    
                    for idx in indices:
                        img_np = 255. * video_tensor[idx].cpu().numpy()
                        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        images_b64.append(img_bytes)
                    print(f"[BSAI_MultiImageVideoReverse] 成功处理视频采样")
                else:
                    print(f"[BSAI_MultiImageVideoReverse] WARNING: 视频张量维度不支持: {video_tensor.shape}")
            else:
                print(f"[BSAI_MultiImageVideoReverse] WARNING: 视频不是torch张量，跳过处理")
        except Exception as e:
            error_msg = f"处理视频时出错: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        
        return images_b64

    def multi_media_reverse(self, system, prompt, model, url, custom_model="",
                          image1=None, image2=None, image3=None, image4=None, image5=None,
                          image6=None, image7=None, image8=None, image9=None, image10=None,
                          video1=None, video2=None, video3=None, video4=None, video5=None,
                          video_sample_interval=10, max_frames_per_video=5,
                          temperature=0.7, top_p=0.9, max_tokens=32768):
        
        # 确定使用的模型
        if model == "custom" and custom_model:
            selected_model = custom_model
        else:
            selected_model = model
        
        try:
            client = Client(host=url)
        except Exception as e:
            error_msg = f"创建Ollama客户端失败: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            return error_msg, "", []
        
        # 收集所有图像
        all_images_b64 = []
        
        # 处理图像输入
        images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        for i, img in enumerate(images, 1):
            if img is not None:
                print(f"[BSAI_MultiImageVideoReverse] 处理图像 {i}")
                all_images_b64.extend(self._process_image_tensor(img))
        
        # 处理视频输入
        videos = [video1, video2, video3, video4, video5]
        for i, vid in enumerate(videos, 1):
            if vid is not None:
                print(f"[BSAI_MultiImageVideoReverse] 处理视频 {i}")
                all_images_b64.extend(self._process_video_tensor(vid, video_sample_interval, max_frames_per_video))
        
        print(f"[BSAI_MultiImageVideoReverse] 媒体处理完成，共 {len(all_images_b64)} 个图像/帧")
        
        # 检查是否有图像
        if not all_images_b64:
            error_msg = "没有有效的图像或视频输入，请至少连接一张图像或一个视频"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            return error_msg, "", []
        
        # 构建请求选项
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens
        }
        
        print(f"[BSAI_MultiImageVideoReverse] 发送请求到Ollama，模型: {selected_model}")
        
        # 发送请求到Ollama
        try:
            response = client.generate(
                model=selected_model,
                system=system,
                prompt=prompt,
                images=all_images_b64,
                options=options,
                keep_alive="5m"
            )
            
            print(f"[BSAI_MultiImageVideoReverse] 收到Ollama响应")
            
            # 提取响应
            ollama_response_text = response.get('response', '')
            ollama_response_thinking = response.get('thinking', '') if 'thinking' in response else ""
            
            print(f"[BSAI_MultiImageVideoReverse] 响应长度: {len(ollama_response_text)}")
            
            prompt_list = [item.strip() for item in ollama_response_text.split('\n') if item.strip()]
            
            return ollama_response_text, ollama_response_thinking, prompt_list
            
        except Exception as e:
            error_msg = f"Ollama API调用失败: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg, "", []

NODE_CLASS_MAPPINGS = {
    "BSAI_MultiImageVideoReverse": BSAI_MultiImageVideoReverse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MultiImageVideoReverse": "BSAI 多图像视频反推",
}
