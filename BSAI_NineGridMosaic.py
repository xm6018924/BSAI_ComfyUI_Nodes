# -*- coding: utf-8 -*-
"""
BSAI_NineGridMosaic
====================
BSAI 原生九宫格拼图节点。

功能对标社区 NineGridMosaicNode（未公开插件）：
- 最多 9 张图像输入（图像1~图像9），按 行数×列数 拼成网格
- 支持接缝（间隙）颜色 / 宽度、背景色、随机排序（带种子）
- 每格等比 contain 缩放并居中，不拉伸变形

与 BSAI_MergeImages 的区别：
- 本节点为 9 个独立 IMAGE 端口（逐个接线），适合替代工作流中的 NineGridMosaicNode
- BSAI_MergeImages 为批量 images 列表输入 + layout 预设，适合循环/列表场景

注册类名：BSAI_NineGridMosaic
类别：BSAI/Image
"""

import random

import numpy as np
import torch
from PIL import Image


def _hex_to_rgb(color_str, default=(255, 255, 255)):
    """将 '#rrggbb' 转为 (r,g,b)；非法值回退默认白色。"""
    if isinstance(color_str, (list, tuple)) and color_str:
        color_str = color_str[0]
    if not isinstance(color_str, str):
        return default
    s = color_str.strip().lstrip('#')
    try:
        if len(s) == 6:
            return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
        if len(s) == 3:
            return tuple(int(s[i] * 2, 16) for i in (0, 1, 2))
    except ValueError:
        pass
    return default


class BSAI_NineGridMosaic:
    """
    九宫格马赛克拼图节点：
    把 1~9 张图像按 行数×列数 排成网格，可自定义接缝（间隙）颜色/宽度、
    背景颜色，并支持按随机种子打乱顺序。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "rows": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1,
                                   "tooltip": "Grid row count (1-16)"}),
                "cols": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                                   "tooltip": "Grid column count (1-16)"}),
                "gap_color": ("COLOR", {"default": "#ffffff", "tooltip": "Color of the gaps between cells"}),
                "gap_width": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1,
                                       "tooltip": "Gap width between cells (pixels); 0 = seamless"}),
                "shuffle": ("BOOLEAN", {"default": False, "tooltip": "Whether to shuffle the image order before merging"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1,
                                       "tooltip": "Seed for the shuffle"}),
                "bg_color": ("COLOR", {"default": "#ffffff", "tooltip": "Background color for empty cells and the border"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_result",)
    OUTPUT_NODE = False
    FUNCTION = "merge"
    CATEGORY = "BSAI/Image"
    DESCRIPTION = "BSAI nine-grid mosaic: merge 1-9 images into a grid, with gap color/width, shuffle and background color support"

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _flatten_images(input_data):
        """把任意嵌套结构压平成 PIL RGB 图像列表（兼容 torch 张量/单图/列表）。"""
        result = []

        def _recurse(item, depth=0):
            if depth > 10:
                return
            if isinstance(item, torch.Tensor):
                if item.dim() == 4:
                    for j in range(item.shape[0]):
                        _recurse(item[j], depth + 1)
                elif item.dim() == 3:
                    result.append(BSAI_NineGridMosaic._tensor_to_pil(item.detach().cpu()))
                elif item.dim() == 2:
                    result.append(BSAI_NineGridMosaic._tensor_to_pil(
                        item.detach().cpu().unsqueeze(-1).repeat(1, 1, 3)))
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    _recurse(sub, depth + 1)
            elif isinstance(item, Image.Image):
                result.append(item.convert("RGB"))

        _recurse(input_data)
        return result

    @staticmethod
    def _tensor_to_pil(img_tensor):
        arr = img_tensor.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        return pil.convert("RGB")

    @staticmethod
    def _fit_contain(img, cell_w, cell_h, bg_rgb):
        """等比缩放并居中放入 cell，比例保持，四周用背景色填充。"""
        img = img.copy()
        img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_w, cell_h), bg_rgb)
        x = (cell_w - img.width) // 2
        y = (cell_h - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas

    @staticmethod
    def _unwrap(value, default):
        """widget 值可能以 [x] 列表形式传入，取第一个有效值。"""
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value if value is not None else default

    # ------------------------------------------------------------------ main
    def merge(self, image_1, image_2=None, image_3=None, image_4=None, image_5=None,
              image_6=None, image_7=None, image_8=None, image_9=None,
              rows=1, cols=4, gap_color="#ffffff", gap_width=0,
              shuffle=False, seed=0, bg_color="#ffffff",
              unique_id=None):
        # 解包 widget 值（兼容 list 包裹）
        rows = int(self._unwrap(rows, 1))
        cols = int(self._unwrap(cols, 4))
        gap_width = int(self._unwrap(gap_width, 0))
        shuffle = bool(self._unwrap(shuffle, False))
        seed = int(self._unwrap(seed, 0))
        gap_rgb = _hex_to_rgb(gap_color)
        bg_rgb = _hex_to_rgb(bg_color)

        rows = max(1, min(16, rows))
        cols = max(1, min(16, cols))
        gap_width = max(0, min(128, gap_width))

        # 收集图像
        all_images = []
        for img in (image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9):
            if img is None:
                continue
            all_images.extend(self._flatten_images(img))
        if not all_images:
            raise ValueError("[BSAI_NineGridMosaic] 至少需要一张图像输入")

        # 随机排序
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(all_images)

        # 只保留 行×列 以内的图像，多余丢弃
        capacity = rows * cols
        images = all_images[:capacity]

        # cell 尺寸 = 所有输入图的最大宽高（保证格子统一）
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)

        # 输出画布尺寸：内部缝隙用接缝颜色，四周留 gap_width 的边框用背景色
        out_w = cell_w * cols + gap_width * (cols - 1) + gap_width * 2
        out_h = cell_h * rows + gap_width * (rows - 1) + gap_width * 2
        canvas = Image.new("RGB", (out_w, out_h), bg_rgb)

        # 填入每一格
        for idx in range(capacity):
            row, col = divmod(idx, cols)
            x = gap_width + col * (cell_w + gap_width)
            y = gap_width + row * (cell_h + gap_width)
            if idx < len(images):
                cell = self._fit_contain(images[idx], cell_w, cell_h, bg_rgb)
                canvas.paste(cell, (x, y))
            # 空格保持背景色

        # 内部缝隙上色：在每行/每列间隙画接缝色条（若 gap>0）
        if gap_width > 0:
            draw = _ImageDraw()
            for r in range(rows - 1):
                y0 = gap_width + (r + 1) * cell_h + r * gap_width
                draw.rectangle(canvas, (0, y0, out_w, y0 + gap_width), gap_rgb)
            for c in range(cols - 1):
                x0 = gap_width + (c + 1) * cell_w + c * gap_width
                draw.rectangle(canvas, (x0, 0, x0 + gap_width, out_h), gap_rgb)

        # 转 torch 张量输出
        merged_np = np.asarray(canvas, dtype=np.float32) / 255.0
        merged_tensor = torch.from_numpy(merged_np).unsqueeze(0).contiguous()
        print(f"[BSAI_NineGridMosaic] grid={rows}x{cols}, "
              f"images={len(images)}, output={merged_tensor.shape}, "
              f"gap={gap_width}px")
        return (merged_tensor,)


class _ImageDraw:
    """极简矩形填充（避免依赖 ImageDraw 时的兼容问题）。"""

    @staticmethod
    def rectangle(canvas, box, rgb):
        from PIL import ImageDraw
        ImageDraw.Draw(canvas).rectangle(box, fill=rgb)


NODE_CLASS_MAPPINGS = {
    "BSAI_NineGridMosaic": BSAI_NineGridMosaic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_NineGridMosaic": "BSAI Nine-Grid Mosaic",
}
