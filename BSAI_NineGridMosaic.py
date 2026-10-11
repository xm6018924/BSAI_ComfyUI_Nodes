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
                "图像1": ("IMAGE",),
            },
            "optional": {
                "图像2": ("IMAGE",),
                "图像3": ("IMAGE",),
                "图像4": ("IMAGE",),
                "图像5": ("IMAGE",),
                "图像6": ("IMAGE",),
                "图像7": ("IMAGE",),
                "图像8": ("IMAGE",),
                "图像9": ("IMAGE",),
                "行数": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1,
                                   "tooltip": "网格行数（1~16）"}),
                "列数": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                                   "tooltip": "网格列数（1~16）"}),
                "接缝颜色": ("COLOR", {"default": "#ffffff", "tooltip": "格子之间的缝隙颜色"}),
                "接缝宽度": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1,
                                       "tooltip": "格子之间的缝隙宽度（像素），0 表示无缝"}),
                "随机排序": ("BOOLEAN", {"default": False, "tooltip": "是否随机打乱图像顺序再拼图"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1,
                                       "tooltip": "随机排序的种子"}),
                "背景颜色": ("COLOR", {"default": "#ffffff", "tooltip": "空白格与四周背景颜色"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("九宫格结果",)
    OUTPUT_NODE = False
    FUNCTION = "merge"
    CATEGORY = "BSAI/Image"
    DESCRIPTION = "BSAI 九宫格马赛克拼图：1~9 张图按行列拼网格，支持接缝颜色/宽度、随机排序、背景色"

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
    def merge(self, 图像1, 图像2=None, 图像3=None, 图像4=None, 图像5=None,
              图像6=None, 图像7=None, 图像8=None, 图像9=None,
              行数=1, 列数=4, 接缝颜色="#ffffff", 接缝宽度=0,
              随机排序=False, 随机种子=0, 背景颜色="#ffffff",
              unique_id=None):
        # 解包 widget 值（兼容 list 包裹）
        行数 = int(self._unwrap(行数, 1))
        列数 = int(self._unwrap(列数, 4))
        接缝宽度 = int(self._unwrap(接缝宽度, 0))
        随机排序 = bool(self._unwrap(随机排序, False))
        随机种子 = int(self._unwrap(随机种子, 0))
        gap_rgb = _hex_to_rgb(接缝颜色)
        bg_rgb = _hex_to_rgb(背景颜色)

        行数 = max(1, min(16, 行数))
        列数 = max(1, min(16, 列数))
        接缝宽度 = max(0, min(128, 接缝宽度))

        # 收集图像
        all_images = []
        for img in (图像1, 图像2, 图像3, 图像4, 图像5, 图像6, 图像7, 图像8, 图像9):
            if img is None:
                continue
            all_images.extend(self._flatten_images(img))
        if not all_images:
            raise ValueError("[BSAI_NineGridMosaic] 至少需要一张图像输入")

        # 随机排序
        if 随机排序:
            rng = random.Random(随机种子)
            rng.shuffle(all_images)

        # 只保留 行×列 以内的图像，多余丢弃
        capacity = 行数 * 列数
        images = all_images[:capacity]

        # cell 尺寸 = 所有输入图的最大宽高（保证格子统一）
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)

        # 输出画布尺寸：内部缝隙用接缝颜色，四周留 gap_width 的边框用背景色
        out_w = cell_w * 列数 + 接缝宽度 * (列数 - 1) + 接缝宽度 * 2
        out_h = cell_h * 行数 + 接缝宽度 * (行数 - 1) + 接缝宽度 * 2
        canvas = Image.new("RGB", (out_w, out_h), bg_rgb)

        # 填入每一格
        for idx in range(capacity):
            row, col = divmod(idx, 列数)
            x = 接缝宽度 + col * (cell_w + 接缝宽度)
            y = 接缝宽度 + row * (cell_h + 接缝宽度)
            if idx < len(images):
                cell = self._fit_contain(images[idx], cell_w, cell_h, bg_rgb)
                canvas.paste(cell, (x, y))
            # 空格保持背景色

        # 内部缝隙上色：在每行/每列间隙画接缝色条（若 gap>0）
        if 接缝宽度 > 0:
            draw = _ImageDraw()
            for r in range(行数 - 1):
                y0 = 接缝宽度 + (r + 1) * cell_h + r * 接缝宽度
                draw.rectangle(canvas, (0, y0, out_w, y0 + 接缝宽度), gap_rgb)
            for c in range(列数 - 1):
                x0 = 接缝宽度 + (c + 1) * cell_w + c * 接缝宽度
                draw.rectangle(canvas, (x0, 0, x0 + 接缝宽度, out_h), gap_rgb)

        # 转 torch 张量输出
        merged_np = np.asarray(canvas, dtype=np.float32) / 255.0
        merged_tensor = torch.from_numpy(merged_np).unsqueeze(0).contiguous()
        print(f"[BSAI_NineGridMosaic] grid={行数}x{列数}, "
              f"images={len(images)}, output={merged_tensor.shape}, "
              f"gap={接缝宽度}px")
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
    "BSAI_NineGridMosaic": "BSAI 九宫格拼图",
}
