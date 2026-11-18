# -*- coding: utf-8 -*-
"""
BSAI_NineGridMosaic
====================
BSAI 原生九宫格拼图节点，并接管社区同名节点类型 `NineGridMosaicNode`。

功能：
- 最多 9 张图像输入，按 行数×列数 拼成网格；每格等比 contain 缩放并居中，不拉伸变形
- 接缝（间隙）颜色 / 宽度、背景色、随机排序（带种子）
- `NineGridMosaicNode` 兼容别名：9 个中文端口（图像1~图像9）+ 输出「九宫格结果」，
  控件顺序与原节点一致（行数/列数/接缝颜色/接缝宽度/随机排序/随机种子/背景色），
  旧工作流无需改线即可直接加载运行；追加「排布模式」控件，默认按接线张数自动排布
  （9 张 → 3×3），需要自定义网格时切到 manual。

与 BSAI_MergeImages 的区别：
- 本节点为 9 个独立 IMAGE 端口（逐个接线），适合替代工作流中的 NineGridMosaicNode
- BSAI_MergeImages 为批量 images 列表输入 + layout 预设，适合循环/列表场景

注册类名：BSAI_NineGridMosaic、NineGridMosaicNode
类别：BSAI/Image
"""

import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

LAYOUT_AUTO = "auto"
LAYOUT_MANUAL = "manual"


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


def _auto_grid(count):
    """按图片张数推算最紧凑网格：1→1×1，2→1×2，3→1×3，4→2×2，5~6→2×3，7~9→3×3。"""
    if count <= 0:
        return 1, 1
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / float(cols)))
    return rows, cols


def _unwrap(value, default):
    """控件值可能以 [x] 列表形式传入，取第一个有效值。"""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


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
                result.append(_tensor_to_pil(item.detach().cpu()))
            elif item.dim() == 2:
                result.append(_tensor_to_pil(
                    item.detach().cpu().unsqueeze(-1).repeat(1, 1, 3)))
        elif isinstance(item, (list, tuple)):
            for sub in item:
                _recurse(sub, depth + 1)
        elif isinstance(item, Image.Image):
            result.append(item.convert("RGB"))

    _recurse(input_data)
    return result


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
    return Image.fromarray(arr, mode="RGB").convert("RGB")


def _fit_contain(img, cell_w, cell_h, bg_rgb):
    """等比缩放并居中放入 cell，比例保持，四周用背景色填充。"""
    img = img.copy()
    img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell_w, cell_h), bg_rgb)
    x = (cell_w - img.width) // 2
    y = (cell_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _mosaic_merge(slots, rows, cols, gap_color, gap_width, shuffle, seed,
                  bg_color, layout_mode=LAYOUT_MANUAL, tag="BSAI_NineGridMosaic"):
    """九宫格拼图核心实现，两套节点注册共用。"""
    rows_in = int(_unwrap(rows, 1))
    cols_in = int(_unwrap(cols, 1))
    gap_width = int(_unwrap(gap_width, 0))
    shuffle = bool(_unwrap(shuffle, False))
    seed = int(_unwrap(seed, 0))
    gap_rgb = _hex_to_rgb(gap_color)
    bg_rgb = _hex_to_rgb(bg_color)
    layout_mode = str(_unwrap(layout_mode, LAYOUT_MANUAL)).strip().lower()

    # 收集图像
    all_images = []
    wired = 0
    for img in slots:
        if img is None:
            continue
        wired += 1
        all_images.extend(_flatten_images(img))
    if not all_images:
        raise ValueError(f"[{tag}] At least one image input is required")

    # 随机排序
    if shuffle:
        random.Random(seed).shuffle(all_images)

    # 网格尺寸
    if layout_mode.startswith("auto"):
        rows, cols = _auto_grid(len(all_images))
        auto_note = f", auto from {len(all_images)} images (ignored rows={rows_in}, cols={cols_in})"
    else:
        rows = max(1, min(16, rows_in))
        cols = max(1, min(16, cols_in))
        auto_note = ""

    capacity = rows * cols
    images = all_images[:capacity]
    dropped = len(all_images) - len(images)

    # cell 尺寸 = 所有参与拼接的图像最大宽高（保证格子统一）
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)

    # 画布：四周 gap_width 边框用背景色；格与格之间 gap_width 用接缝色
    out_w = cell_w * cols + gap_width * (cols + 1)
    out_h = cell_h * rows + gap_width * (rows + 1)
    canvas = Image.new("RGB", (out_w, out_h), bg_rgb)

    for idx in range(capacity):
        row, col = divmod(idx, cols)
        x = gap_width + col * (cell_w + gap_width)
        y = gap_width + row * (cell_h + gap_width)
        if idx < len(images):
            canvas.paste(_fit_contain(images[idx], cell_w, cell_h, bg_rgb), (x, y))
        # 空格保持背景色

    # 内部接缝上色：只画在内容区之内，不覆盖四周背景色边框
    if gap_width > 0:
        draw = ImageDraw.Draw(canvas)
        x0, y0 = gap_width, gap_width
        x1, y1 = out_w - gap_width - 1, out_h - gap_width - 1
        for r in range(rows - 1):
            y = gap_width + (r + 1) * cell_h + r * gap_width
            draw.rectangle([x0, y, x1, y + gap_width - 1], fill=gap_rgb)
        for c in range(cols - 1):
            x = gap_width + (c + 1) * cell_w + c * gap_width
            draw.rectangle([x, y0, x + gap_width - 1, y1], fill=gap_rgb)

    merged_np = np.asarray(canvas, dtype=np.float32) / 255.0
    merged_tensor = torch.from_numpy(merged_np).unsqueeze(0).contiguous()
    print(f"[{tag}] grid={rows}x{cols}, wired={wired}, images={len(images)}, "
          f"cell={cell_w}x{cell_h}, gap={gap_width}px, output={tuple(merged_tensor.shape)}"
          + (f", dropped={dropped}" if dropped else "") + auto_note)
    return (merged_tensor,)


class BSAI_NineGridMosaic:
    """
    九宫格马赛克拼图节点（BSAI 原生）：
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

    def merge(self, image_1, image_2=None, image_3=None, image_4=None, image_5=None,
              image_6=None, image_7=None, image_8=None, image_9=None,
              rows=1, cols=4, gap_color="#ffffff", gap_width=0,
              shuffle=False, seed=0, bg_color="#ffffff",
              unique_id=None):
        return _mosaic_merge(
            (image_1, image_2, image_3, image_4, image_5,
             image_6, image_7, image_8, image_9),
            rows, cols, gap_color, gap_width, shuffle, seed, bg_color,
            LAYOUT_MANUAL, "BSAI_NineGridMosaic",
        )


class NineGridMosaicNode:
    """
    NineGridMosaicNode 完整实现（接管同名节点类型）。

    端口与控件顺序与原节点一致：9 个中文图像端口 + 「九宫格结果」输出；
    控件依次为 行数 / 列数 / 接缝颜色 / 接缝宽度 / 随机排序 / 随机种子 / 背景色，
    因此旧工作流里的 7 个已存值会按位置落到对应参数上。
    末尾追加「排布模式」（默认 auto：按实际接线张数自动排布，9 张 → 3×3）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "图像1": ("IMAGE",),
                "图像2": ("IMAGE",),
                "图像3": ("IMAGE",),
                "图像4": ("IMAGE",),
                "图像5": ("IMAGE",),
                "图像6": ("IMAGE",),
                "图像7": ("IMAGE",),
                "图像8": ("IMAGE",),
                "图像9": ("IMAGE",),
                "行数": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1,
                                  "tooltip": "网格行数；仅「排布模式=manual」时生效"}),
                "列数": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1,
                                  "tooltip": "网格列数；仅「排布模式=manual」时生效"}),
                "接缝颜色": ("COLOR", {"default": "#ffffff", "tooltip": "格子之间接缝的颜色"}),
                "接缝宽度": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1,
                                      "tooltip": "接缝宽度（像素）；0 = 无缝"}),
                "随机排序": ("BOOLEAN", {"default": False, "tooltip": "是否按种子打乱图片顺序"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1,
                                      "tooltip": "随机排序使用的种子"}),
                "背景色": ("COLOR", {"default": "#ffffff", "tooltip": "空格子与四周边框的颜色"}),
                "排布模式": (["auto", "manual"], {"default": "auto",
                              "tooltip": "auto=按接线张数自动排布（9张→3×3）；manual=用行数/列数"}),
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
    DESCRIPTION = "Nine-grid mosaic drop-in (BSAI implementation): 9 image inputs -> one grid image, auto layout by default"

    def merge(self, 图像1=None, 图像2=None, 图像3=None, 图像4=None, 图像5=None,
              图像6=None, 图像7=None, 图像8=None, 图像9=None,
              行数=3, 列数=3, 接缝颜色="#ffffff", 接缝宽度=0,
              随机排序=False, 随机种子=0, 背景色="#ffffff",
              排布模式="auto", unique_id=None):
        return _mosaic_merge(
            (图像1, 图像2, 图像3, 图像4, 图像5, 图像6, 图像7, 图像8, 图像9),
            行数, 列数, 接缝颜色, 接缝宽度, 随机排序, 随机种子, 背景色,
            排布模式, "NineGridMosaicNode",
        )


NODE_CLASS_MAPPINGS = {
    "BSAI_NineGridMosaic": BSAI_NineGridMosaic,
    # 接管社区同名节点类型：旧工作流保存的 NineGridMosaicNode 直接解析到本实现
    "NineGridMosaicNode": NineGridMosaicNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_NineGridMosaic": "BSAI Nine-Grid Mosaic",
    "NineGridMosaicNode": "NineGridMosaicNode",
}
