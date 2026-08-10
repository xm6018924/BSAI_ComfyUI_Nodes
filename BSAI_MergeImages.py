import torch
import numpy as np
import os
from PIL import Image
from datetime import datetime


class BSAI_MergeImages:
    """
    Merge a batch of images into a single large image (full-frame grid).
    Supports both:
    1. INPUT_IS_LIST mode (ComfyUI native list collection)
    2. Accumulation mode (for graph expansion loops like ComfyUI-Easy-Use)
    """

    INPUT_IS_LIST = True
    _accumulated = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "layout": (["Auto", "2x2 (4宫格)", "2x3 (6宫格)", "3x2 (6宫格)", "3x3 (9宫格)", "Custom"], {"default": "Auto", "tooltip": "图像排版方式"}),
            },
            "optional": {
                "custom_rows": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1, "tooltip": "自定义行数（选择Custom时生效）"}),
                "custom_cols": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1, "tooltip": "自定义列数（选择Custom时生效）"}),
                "expected_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "预期图像数量（0表示自动检测）"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "自定义输出宽度，0表示使用原始图像尺寸"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "自定义输出高度，0表示使用原始图像尺寸"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "output_prefix": ("STRING", {"default": "merged"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)

    FUNCTION = "merge_images"
    CATEGORY = "BSAI/Image"

    @staticmethod
    def _flatten_images(input_data):
        result = []

        def _recurse(item, depth=0):
            if depth > 10:
                return

            if isinstance(item, torch.Tensor):
                if item.dim() == 4:
                    for j in range(item.shape[0]):
                        _recurse(item[j], depth + 1)
                elif item.dim() == 3:
                    result.append(item.detach().cpu())
                elif item.dim() == 2:
                    result.append(item.detach().cpu().unsqueeze(-1).repeat(1, 1, 3))
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    _recurse(sub_item, depth + 1)
            elif isinstance(item, Image.Image):
                result.append(item)

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
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return pil

    @staticmethod
    def _get_grid_size(layout, custom_rows, custom_cols):
        if layout == "2x2 (4宫格)":
            return 2, 2
        elif layout == "2x3 (6宫格)":
            return 3, 2
        elif layout == "3x2 (6宫格)":
            return 2, 3
        elif layout == "3x3 (9宫格)":
            return 3, 3
        elif layout == "Custom":
            return max(int(custom_cols), 1), max(int(custom_rows), 1)
        else:
            return 1, 1

    @staticmethod
    def _get_expected_count(layout, custom_rows, custom_cols, expected_count):
        if expected_count > 0:
            return expected_count
        cols, rows = BSAI_MergeImages._get_grid_size(layout, custom_rows, custom_cols)
        return cols * rows

    def _do_merge(self, image_list, layout, custom_rows, custom_cols,
                  width, height, save_output, output_prefix):
        num_images = len(image_list)

        if layout == "Auto":
            columns = int(np.ceil(np.sqrt(num_images)))
            columns = max(columns, 1)
            rows = (num_images + columns - 1) // columns
        elif layout == "2x2 (4宫格)":
            columns, rows = 2, 2
        elif layout == "2x3 (6宫格)":
            columns, rows = 3, 2
        elif layout == "3x2 (6宫格)":
            columns, rows = 2, 3
        elif layout == "3x3 (9宫格)":
            columns, rows = 3, 3
        elif layout == "Custom":
            columns = max(int(custom_cols), 1)
            rows = max(int(custom_rows), 1)
        else:
            columns = int(np.ceil(np.sqrt(num_images)))
            columns = max(columns, 1)
            rows = (num_images + columns - 1) // columns

        # Convert to PIL images
        pil_images = []
        for entry in image_list:
            if isinstance(entry, torch.Tensor):
                pil_images.append(self._tensor_to_pil(entry))
            elif isinstance(entry, Image.Image):
                pil = entry
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                pil_images.append(pil)

        # Calculate output size
        use_custom_size = width > 0 or height > 0
        if use_custom_size:
            if width > 0 and height > 0:
                output_width, output_height = int(width), int(height)
            elif width > 0:
                avg_aspect = sum(im.width / im.height for im in pil_images) / num_images
                output_width = int(width)
                output_height = max(1, int(width / max(columns * avg_aspect, 1e-6) * rows))
            else:
                avg_aspect = sum(im.width / im.height for im in pil_images) / num_images
                output_height = int(height)
                output_width = max(1, int(height * columns * avg_aspect / max(rows, 1)))
        else:
            max_w = max(im.width for im in pil_images)
            max_h = max(im.height for im in pil_images)
            output_width = max_w * columns
            output_height = max_h * rows

        output_width = max(int(output_width), columns)
        output_height = max(int(output_height), rows)

        cell_width = output_width // columns
        cell_height = output_height // rows
        cell_width = max(cell_width, 1)
        cell_height = max(cell_height, 1)

        # Merge images
        merged_image = Image.new("RGB", (output_width, output_height), (255, 255, 255))
        for i, pil_img in enumerate(pil_images):
            if i >= rows * columns:
                break
            row = i // columns
            col = i % columns
            resized = pil_img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            x = col * cell_width
            y = row * cell_height
            merged_image.paste(resized, (x, y))

        # Save
        if save_output:
            try:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                output_dir = os.path.join(project_root, "output")
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{output_prefix}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                merged_image.save(filepath, quality=95)
                print(f"[BSAI_MergeImages] Saved to: {filepath}")
            except Exception as exc:
                print(f"[BSAI_MergeImages] Save failed: {exc}")

        # Convert to tensor
        merged_np = np.asarray(merged_image, dtype=np.float32) / 255.0
        if merged_np.ndim == 2:
            merged_np = np.stack([merged_np] * 3, axis=-1)
        elif merged_np.ndim == 3 and merged_np.shape[-1] == 4:
            merged_np = merged_np[:, :, :3]
        elif merged_np.ndim == 3 and merged_np.shape[-1] == 1:
            merged_np = np.repeat(merged_np, 3, axis=-1)

        merged_tensor = torch.from_numpy(merged_np).unsqueeze(0).contiguous()
        print(f"[BSAI_MergeImages] Output tensor: {merged_tensor.shape}, grid={columns}x{rows}")

        return (merged_tensor,)

    def merge_images(self, images, layout, custom_rows, custom_cols,
                     expected_count, width, height, save_output, output_prefix,
                     unique_id=None):
        """
        images: list of tensors (due to INPUT_IS_LIST=True)
        """
        # Unpack list parameters
        if isinstance(layout, (list, tuple)) and len(layout) > 0:
            layout = layout[0]
        else:
            layout = "Auto"

        if isinstance(custom_rows, (list, tuple)) and len(custom_rows) > 0:
            custom_rows = custom_rows[0]
        else:
            custom_rows = 2

        if isinstance(custom_cols, (list, tuple)) and len(custom_cols) > 0:
            custom_cols = custom_cols[0]
        else:
            custom_cols = 2

        if isinstance(expected_count, (list, tuple)) and len(expected_count) > 0:
            expected_count = expected_count[0]
        else:
            expected_count = 0

        if isinstance(width, (list, tuple)) and len(width) > 0:
            width = width[0]
        else:
            width = 0

        if isinstance(height, (list, tuple)) and len(height) > 0:
            height = height[0]
        else:
            height = 0

        if isinstance(save_output, (list, tuple)) and len(save_output) > 0:
            save_output = save_output[0]
        else:
            save_output = True

        if isinstance(output_prefix, (list, tuple)) and len(output_prefix) > 0:
            output_prefix = output_prefix[0]
        else:
            output_prefix = "merged"

        # Flatten all images from the list
        all_images = []
        for item in images:
            all_images.extend(self._flatten_images(item))

        num_images = len(all_images)
        print(f"[BSAI_MergeImages] Received {num_images} images from list, layout={layout}")

        # If we got multiple images directly (INPUT_IS_LIST worked), merge immediately
        if num_images > 1:
            return self._do_merge(all_images, layout, custom_rows, custom_cols,
                                  width, height, save_output, output_prefix)

        # If only 1 image, we might be in graph expansion loop mode
        # Use accumulation logic
        key = unique_id if unique_id else "default"
        expected = self._get_expected_count(layout, custom_rows, custom_cols, expected_count)

        if key not in BSAI_MergeImages._accumulated:
            BSAI_MergeImages._accumulated[key] = {
                "images": [],
                "params": {
                    "layout": layout,
                    "custom_rows": custom_rows,
                    "custom_cols": custom_cols,
                    "width": width,
                    "height": height,
                    "save_output": save_output,
                    "output_prefix": output_prefix,
                }
            }

        entry = BSAI_MergeImages._accumulated[key]
        entry["images"].extend(all_images)
        current_count = len(entry["images"])

        print(f"[BSAI_MergeImages] Accumulated {current_count}/{expected} images (node={key})")

        if current_count < expected:
            placeholder = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
            return (placeholder,)

        # Merge accumulated images
        all_accumulated = entry["images"][:expected]
        try:
            return self._do_merge(all_accumulated, layout, custom_rows, custom_cols,
                                  width, height, save_output, output_prefix)
        finally:
            BSAI_MergeImages._accumulated.pop(key, None)


NODE_CLASS_MAPPINGS = {
    "BSAI_MergeImages": BSAI_MergeImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MergeImages": "BSAI Merge Images",
}
