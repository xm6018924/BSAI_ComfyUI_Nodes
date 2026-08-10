"""
BSAI Draw Text Overlay Node
===========================
基于 ComfyUI 原生 TextOverlay 节点增强，新增系统字体自定义选择功能，
支持 Windows 系统字体目录中的所有字体（含各国文字与 Unicode 字形）。

特点：
- 可从下拉列表选择系统已安装的任意字体（TrueType / OpenType / TTC）
- 字体目录通过系统环境变量自动定位，不硬编码盘符，便于跨机器/盘符迁移
- 保留原节点全部参数与行为（字号百分比、颜色、位置、对齐、描边）
- 选中字体缺失时自动回退 PIL 默认字体，保证健壮性
"""

import os
import re
import logging
import numpy as np
import torch
from PIL import Image as PILImage, ImageColor, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ===========================================================================
# 系统字体发现
# ===========================================================================

_DEFAULT_FONT_LABEL = "Default (PIL)"

# 模块级缓存：避免每次 INPUT_TYPES 调用都重新扫描注册表/目录
_FONT_LIST_CACHE = None        # list[str]  —— 下拉菜单显示名
_FONT_PATH_MAP_CACHE = None    # dict[str, str] —— 显示名 -> 字体文件绝对路径


def _get_system_fonts_dir():
    """通过系统环境变量定位 Windows 字体目录（不硬编码盘符）。"""
    system_root = os.environ.get('SystemRoot') or os.environ.get('WINDIR') or r'C:\Windows'
    return os.path.join(system_root, 'Fonts')


def _read_registry_fonts():
    """读取 Windows 注册表，获取「字体显示名 -> 文件路径」映射。

    注册表条目形如：
        "Arial (TrueType)"          -> "arial.ttf"
        "Microsoft YaHei (TrueType)" -> "msyh.ttc"
    去掉后缀后得到简洁的显示名。
    """
    import winreg
    mapping = {}
    fonts_dir = _get_system_fonts_dir()
    hives = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
    ]
    for hive, subkey in hives:
        try:
            key = winreg.OpenKey(hive, subkey)
        except OSError:
            continue
        try:
            index = 0
            while True:
                try:
                    name, value, _vtype = winreg.EnumValue(key, index)
                    index += 1
                except OSError:
                    break
                if not value:
                    continue
                # 去掉 " (TrueType)" / " (OpenType)" 等后缀
                display = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
                if not display:
                    continue
                # value 可能是文件名也可能是绝对路径
                path = value if os.path.isabs(value) else os.path.join(fonts_dir, value)
                if os.path.isfile(path) and display not in mapping:
                    mapping[display] = path
        finally:
            winreg.CloseKey(key)
    return mapping


def _enumerate_font_files():
    """回退方案：直接枚举字体目录中的字体文件。"""
    mapping = {}
    fonts_dir = _get_system_fonts_dir()
    if os.path.isdir(fonts_dir):
        for fname in os.listdir(fonts_dir):
            low = fname.lower()
            if low.endswith(('.ttf', '.ttc', '.otf')):
                path = os.path.join(fonts_dir, fname)
                display = os.path.splitext(fname)[0]
                if display not in mapping:
                    mapping[display] = path
    return mapping


def _ensure_font_cache():
    """构建并缓存系统字体列表。

    线程安全说明：缓存写入可能并发，但结果幂等（相同输入产生相同映射），
    无需加锁；最坏情况下多算一次，不影响正确性。
    """
    global _FONT_LIST_CACHE, _FONT_PATH_MAP_CACHE
    if _FONT_LIST_CACHE is not None:
        return

    try:
        mapping = _read_registry_fonts()
        if not mapping:
            mapping = _enumerate_font_files()
    except Exception as e:
        logger.warning(f"[BSAI_DrawTextOverlay] 注册表读取失败，回退文件枚举: {e}")
        try:
            mapping = _enumerate_font_files()
        except Exception:
            mapping = {}

    # 首项为 PIL 默认字体（与原节点行为一致，不支持中文等复杂字形）
    path_map = {_DEFAULT_FONT_LABEL: ""}
    for name in sorted(mapping.keys(), key=lambda x: x.lower()):
        path_map[name] = mapping[name]

    _FONT_PATH_MAP_CACHE = path_map
    _FONT_LIST_CACHE = list(path_map.keys())


def get_font_choices():
    """返回下拉菜单使用的字体显示名列表。"""
    _ensure_font_cache()
    return _FONT_LIST_CACHE if _FONT_LIST_CACHE else [_DEFAULT_FONT_LABEL]


def resolve_font_path(display_name):
    """根据显示名解析字体文件绝对路径，找不到时返回空串（回退默认字体）。"""
    _ensure_font_cache()
    if _FONT_PATH_MAP_CACHE is None:
        return ""
    return _FONT_PATH_MAP_CACHE.get(display_name, "")


# ===========================================================================
# 节点
# ===========================================================================

class BSAI_DrawTextOverlay:
    """
    在图像上绘制文字覆盖层（增强版）。
    支持自定义选择 Windows 系统字体，覆盖各国文字与 Unicode 字形。
    """

    @classmethod
    def INPUT_TYPES(cls):
        font_choices = get_font_choices()
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "font_name": (font_choices, {
                    "default": font_choices[0] if font_choices else _DEFAULT_FONT_LABEL,
                    "tooltip": "选择系统字体。列表来自 Windows 字体目录，支持各国文字。",
                }),
                "font_size": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 50.0, "step": 0.5,
                    "tooltip": "字号，按图像高度的百分比计算。",
                }),
                "color": ("COLOR", {"default": "#ffffff", "tooltip": "文字颜色。"}),
                "position": (["top", "bottom"], {"default": "top"}),
                "align": (["left", "center", "right"], {"default": "left"}),
                "outline": ("BOOLEAN", {"default": True, "tooltip": "绘制黑色描边以增强可读性。"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "draw_text_overlay"
    CATEGORY = "BSAI/Image"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 允许不在当前系统列表中的字体名通过（运行时自动回退默认字体），
        # 避免工作流跨机器迁移时因字体缺失而无法执行。
        return True

    # ----- 颜色 / 字体辅助 -----

    @staticmethod
    def _parse_color_to_rgba(color_string):
        parsed = ImageColor.getrgb(color_string)
        if len(parsed) == 3:
            return (*parsed, 255)
        return parsed

    @staticmethod
    def _load_font(font_path, size):
        """加载字体：优先 TrueType，失败或无路径时回退 PIL 默认字体。"""
        if font_path and os.path.isfile(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception as e:
                logger.warning(f"[BSAI_DrawTextOverlay] 字体加载失败 '{font_path}': {e}，回退默认字体")
        return ImageFont.load_default(size=size)

    # ----- 文本换行（与原节点逻辑一致，支持 CJK 逐字断行） -----

    @staticmethod
    def _wrap_text(text, font, max_width):
        lines = []
        for raw_line in text.split("\n"):
            words = raw_line.split()
            if not words:
                lines.append("")
                continue
            current = ""
            for word in words:
                while font.getlength(word) > max_width and len(word) > 1:
                    cut = 1
                    while cut < len(word) and font.getlength(word[:cut + 1]) <= max_width:
                        cut += 1
                    if current:
                        lines.append(current)
                        current = ""
                    lines.append(word[:cut])
                    word = word[cut:]
                candidate = word if not current else current + " " + word
                if not current or font.getlength(candidate) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
        return lines

    # ----- 核心渲染 -----

    @classmethod
    def _render_overlay_text(cls, width, height, text, position, align,
                             font_size, text_rgba, outline_rgba, font_path):
        line_spacing = 1.2
        margin_percent = 1.0
        min_font_percent = 2.0
        min_font_pixels = 10
        outline_thickness_factor = 0.04

        # 在透明图层上绘制，便于后续 alpha 合成到任意帧
        layer = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        margin = int(round(margin_percent / 100.0 * min(width, height)))
        max_width = max(1, width - 2 * margin)
        max_height = max(1, height - 2 * margin)

        # 字号随分辨率缩放，再逐步缩小以适配高度
        size = max(1, int(round(font_size / 100.0 * height)))
        floor = min(size, max(min_font_pixels, int(round(min_font_percent / 100.0 * height))))

        font = None
        block = ""
        box = (0, 0, 0, 0)
        stroke = 0
        pixel_spacing = 0

        while True:
            font = cls._load_font(font_path, size)
            stroke = max(1, int(round(size * outline_thickness_factor))) if outline_rgba[3] > 0 else 0
            block = "\n".join(cls._wrap_text(text, font, max_width))
            # 行距换算为像素间距
            single = draw.textbbox((0, 0), "Ay", font=font, stroke_width=stroke)
            double = draw.multiline_textbbox((0, 0), "Ay\nAy", font=font, spacing=0, stroke_width=stroke)
            natural_advance = (double[3] - double[1]) - (single[3] - single[1])
            pixel_spacing = int(round(size * line_spacing - natural_advance))
            box = draw.multiline_textbbox((0, 0), block, font=font, spacing=pixel_spacing, stroke_width=stroke)
            block_height = box[3] - box[1]

            if block_height <= max_height or size <= floor:
                break
            size = max(floor, int(size * 0.9))

        anchor_h, x = {
            "left": ("l", margin),
            "center": ("m", width / 2),
            "right": ("r", width - margin),
        }[align]

        # 偏移 y 使文字紧贴边距
        if position == "bottom":
            y = height - margin - box[3]
        else:
            y = margin - box[1]

        draw.multiline_text(
            (x, y), block, font=font, fill=text_rgba, anchor=anchor_h + "a",
            align=align, spacing=pixel_spacing, stroke_width=stroke, stroke_fill=outline_rgba,
        )

        overlay = np.array(layer).astype(np.float32) / 255.0
        overlay_rgb = torch.from_numpy(overlay[:, :, :3])
        overlay_alpha = torch.from_numpy(overlay[:, :, 3:4])
        return overlay_rgb, overlay_alpha

    # ----- 入口 -----

    def draw_text_overlay(self, images, text, font_name, font_size, color, position, align, outline):
        if text.strip() == "":
            return (images,)

        text = text.replace("\\n", "\n").replace("\\t", "\t")

        text_rgba = self._parse_color_to_rgba(color)
        outline_rgba = (0, 0, 0, 255) if outline else (0, 0, 0, 0)

        font_path = resolve_font_path(font_name)

        height = images.shape[1]
        width = images.shape[2]
        overlay_rgb, overlay_alpha = self._render_overlay_text(
            width, height, text, position, align,
            font_size, text_rgba, outline_rgba, font_path,
        )
        overlay_rgb = overlay_rgb.to(device=images.device, dtype=images.dtype)
        overlay_alpha = overlay_alpha.to(device=images.device, dtype=images.dtype)

        result = images * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
        return (result,)


NODE_CLASS_MAPPINGS = {
    "BSAI_DrawTextOverlay": BSAI_DrawTextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_DrawTextOverlay": "BSAI Draw Text Overlay",
}
