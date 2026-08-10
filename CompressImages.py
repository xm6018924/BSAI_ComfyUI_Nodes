import torch
import numpy as np
from PIL import Image
import io
import base64
from comfy import utils

class CompressImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images or video_path": ("IMAGE",),
                "format": (["JPEG", "PNG", "WEBP"], {"default": "JPEG"}),
                "quality": ("INT", {
                    "default": 85,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Compression quality (1-100). Higher values mean better quality but larger file size."
                }),
            },
            "optional": {
                "compress_level": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 9,
                    "step": 1,
                    "tooltip": "Compression level for PNG (0-9). Higher values mean better compression but slower."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "compressed_data",)
    FUNCTION = "compress_images"
    CATEGORY = "BSAI/image"
    DESCRIPTION = "Compress images to reduce file size while maintaining quality"

    def compress_images(self, **kwargs):
        # 处理参数，支持两种输入名称
        images = kwargs.get("images or video_path", kwargs.get("images"))
        format = kwargs.get("format", "JPEG")
        quality = kwargs.get("quality", 85)
        compress_level = kwargs.get("compress_level", 4)
        
        compressed_images = []
        compressed_data_list = []
        
        for i in range(images.shape[0]):
            img_tensor = images[i]
            img_np = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            
            # 清理中间变量
            del img_np, img_tensor
            import gc
            gc.collect()
            
            buffered = io.BytesIO()
            
            if format == "JPEG":
                img.save(buffered, format="JPEG", quality=quality, optimize=True)
            elif format == "PNG":
                img.save(buffered, format="PNG", compress_level=compress_level, optimize=True)
            elif format == "WEBP":
                img.save(buffered, format="WEBP", quality=quality, method=6)
            
            # 清理图像对象
            del img
            gc.collect()
            
            compressed_bytes = buffered.getvalue()
            compressed_data_list.append(base64.b64encode(compressed_bytes).decode("utf-8"))
            
            # 将压缩后的图像转换回tensor
            compressed_img = Image.open(io.BytesIO(compressed_bytes))
            if compressed_img.mode != 'RGB':
                compressed_img = compressed_img.convert('RGB')
            compressed_np = np.array(compressed_img).astype(np.float32) / 255.0
            compressed_tensor = torch.from_numpy(compressed_np).unsqueeze(0)
            compressed_images.append(compressed_tensor)
            
            # 清理中间变量
            del compressed_img, compressed_np, compressed_bytes, buffered
            gc.collect()
        
        result = torch.cat(compressed_images, dim=0)
        
        # 清理列表
        del compressed_images
        gc.collect()
        
        return (result, "\n".join(compressed_data_list),)

NODE_CLASS_MAPPINGS = {
    "CompressImages": CompressImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompressImages": "Compress Images",
}