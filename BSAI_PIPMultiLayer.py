import torch
import numpy as np
import cv2
from PIL import Image
import folder_paths

class BSAI_PIPMultiLayerWithAudio:
    """
    BSAI 多视频画中画叠加节点：
    - 多视频输入端口（可扩展）
    - 每层 X/Y/Width/Height 自定义
    - 多音频输入 + 选择输出哪一路音频
    - 输出：合成视频帧 + 选中音频
    """
    CATEGORY = "BSAI/Video"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("composite_image", "selected_audio")
    FUNCTION = "process_pip"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 基础画布大小
                "canvas_width": ("INT", {"default": 1920, "min": 256, "max": 7680, "step": 2}),
                "canvas_height": ("INT", {"default": 1080, "min": 256, "max": 4320, "step": 2}),
                # 音频选择
                "audio_select": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            },
            "optional": {
                # 视频层 0（底层/背景）
                "video_0": ("IMAGE",),
                "x0": ("INT", {"default": 0, "min": 0, "step": 1}),
                "y0": ("INT", {"default": 0, "min": 0, "step": 1}),
                "w0": ("INT", {"default": 1920, "min": 1, "step": 2}),
                "h0": ("INT", {"default": 1080, "min": 1, "step": 2}),

                # 视频层 1（画中画1）
                "video_1": ("IMAGE",),
                "x1": ("INT", {"default": 100, "min": 0, "step": 1}),
                "y1": ("INT", {"default": 100, "min": 0, "step": 1}),
                "w1": ("INT", {"default": 640, "min": 1, "step": 2}),
                "h1": ("INT", {"default": 360, "min": 1, "step": 2}),

                # 视频层 2（画中画2）
                "video_2": ("IMAGE",),
                "x2": ("INT", {"default": 800, "min": 0, "step": 1}),
                "y2": ("INT", {"default": 100, "min": 0, "step": 1}),
                "w2": ("INT", {"default": 640, "min": 1, "step": 2}),
                "h2": ("INT", {"default": 360, "min": 1, "step": 2}),

                # 视频层 3（画中画3）
                "video_3": ("IMAGE",),
                "x3": ("INT", {"default": 100, "min": 0, "step": 1}),
                "y3": ("INT", {"default": 500, "min": 0, "step": 1}),
                "w3": ("INT", {"default": 640, "min": 1, "step": 2}),
                "h3": ("INT", {"default": 360, "min": 1, "step": 2}),

                # 音频
                "audio_0": ("AUDIO",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
            }
        }

    def process_pip(self, canvas_width, canvas_height, audio_select,
                   video_0=None, x0=0, y0=0, w0=0, h0=0,
                   video_1=None, x1=0, y1=0, w1=0, h1=0,
                   video_2=None, x2=0, y2=0, w2=0, h2=0,
                   video_3=None, x3=0, y3=0, w3=0, h3=0,
                   audio_0=None, audio_1=None, audio_2=None, audio_3=None):

        # 收集所有层
        layers = [
            (video_0, x0, y0, w0, h0),
            (video_1, x1, y1, w1, h1),
            (video_2, x2, y2, w2, h2),
            (video_3, x3, y3, w3, h3),
        ]
        audios = [audio_0, audio_1, audio_2, audio_3]

        # 选中音频
        out_audio = audios[audio_select] if audio_select < len(audios) and audios[audio_select] is not None else None

        # 取第0路视频帧数作为总帧数（对齐长度）
        frame_count = 0
        for vid, _, _, _, _ in layers:
            if vid is not None:
                frame_count = vid.shape[0]
                break
        if frame_count == 0:
            raise Exception("至少需要一路视频输入")

        composite_frames = []

        for idx in range(frame_count):
            # 新建黑色画布
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)

            # 逐层叠加
            for vid, x, y, w, h in layers:
                if vid is None:
                    continue
                if idx >= vid.shape[0]:
                    continue  # 帧数不足则跳过

                # 转 numpy HWC 0~1
                frame = vid[idx].cpu().numpy()

                # 缩放到目标大小
                frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                # 计算有效区域（防止越界）
                h_canv, w_canv = canvas.shape[:2]
                x_end = min(x + w, w_canv)
                y_end = min(y + h, h_canv)
                x_src_end = x_end - x
                y_src_end = y_end - y
                if x_src_end <= 0 or y_src_end <= 0:
                    continue

                # 贴到画布
                canvas[y:y_end, x:x_end] = frame_resized[0:y_src_end, 0:x_src_end]

            # 转回 ComfyUI 格式：B,H,W,C / float32 / 0~1
            canvas_tensor = torch.from_numpy(canvas).unsqueeze(0)
            composite_frames.append(canvas_tensor)

        # 拼接成批次
        composite_out = torch.cat(composite_frames, dim=0)

        return (composite_out, out_audio)

# 注册
NODE_CLASS_MAPPINGS = {
    "BSAI_PIPMultiLayerWithAudio": BSAI_PIPMultiLayerWithAudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_PIPMultiLayerWithAudio": "BSAI PIP MultiLayer (Multi Video + Audio Select)"
}