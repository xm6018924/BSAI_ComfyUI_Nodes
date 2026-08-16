import cv2
import os
import folder_paths
import torch
import numpy as np
import gc
import inspect
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.latest import InputImpl
import comfy.model_management as mm

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
except Exception:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except Exception:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
except Exception:
    Gemma4ChatHandler = None

import tempfile
import time
from pathlib import Path
from PIL import Image


class BSAI_MultiplePathsInputPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "path_1": ("PATH",),
            },
            "optional": {
                "sample_fps": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "max_frames": (
                    "INT",
                    {"default": 2, "min": 2, "max": (1 << 63) - 1, "step": 1},
                ),
                "use_total_frames": ("BOOLEAN", {"default": True}),
                "use_original_fps_as_sample_fps": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "combine_plus"
    CATEGORY = "BSAI"
    DESCRIPTION = """
Creates a path batch from multiple paths (BSAI Plus version).
Enhanced with batch processing support.
"""

    @staticmethod
    def convert_path_to_json(
        file_path,
        sample_fps=1,
        max_frames=1,
        use_total_frames=True,
        use_original_fps_as_sample_fps=True,
    ):
        ext = file_path.split(".")[-1].lower()

        if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
            return {"type": "image", "image": f"{file_path}"}
        elif ext in ["mp4", "mkv", "mov", "avi", "flv", "wmv", "webm", "m4v"]:
            vidObj = cv2.VideoCapture(file_path)
            vr = []
            while vidObj.isOpened():
                ret, frame = vidObj.read()
                if not ret:
                    break
                else:
                    vr.append(frame)
            total_frames = len(vr) + 1
            avg_fps = vidObj.get(cv2.CAP_PROP_FPS)
            vidObj.release()
            return {
                "type": "video",
                "video": f"{file_path}",
                "fps": avg_fps if use_original_fps_as_sample_fps else sample_fps,
                "max_frames": total_frames if use_total_frames else max_frames,
            }
        else:
            return None

    def combine_plus(self, inputcount, **kwargs):
        path_list = []
        for c in range(inputcount):
            path = kwargs[f"path_{c + 1}"]

            filtered_kwargs = {
                k: v for k, v in kwargs.items() if not k.startswith("path_") and k != "batch_size"
            }

            path = self.convert_path_to_json(path, **filtered_kwargs)
            path_list.append(path)
        return (path_list,)


class BSAI_VideoLoaderPlus(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "file": (sorted(files), {"video_upload": True}),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": (1 << 63) - 1, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": (1 << 63) - 1, "step": 1}),
            },
            "optional": {
                "loop": ("BOOLEAN", {"default": False}),
                "resize_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "resize_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            },
        }

    CATEGORY = "BSAI"

    RETURN_TYPES = (IO.VIDEO, "PATH", "INT")
    RETURN_NAMES = ("video", "path", "total_frames")
    FUNCTION = "load_video_plus"

    def load_video_plus(self, file, frame_rate=30.0, start_frame=0, end_frame=-1, loop=False, resize_width=0, resize_height=0):
        video_path = folder_paths.get_annotated_filepath(file)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if end_frame < 0:
            end_frame = total_frames
        
        return (InputImpl.VideoFromFile(video_path), video_path, total_frames)

    @classmethod
    def IS_CHANGED(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        return mod_time

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)
        return True


# Qwen Model Storage for BSAI nodes
class _BSAI_QwenStorage:
    model = None
    settings = None

    @classmethod
    def unload(cls):
        try:
            if cls.model and hasattr(cls.model, 'close'):
                cls.model.close()
        except Exception:
            pass
        cls.model = None
        cls.settings = None
        gc.collect()
        mm.soft_empty_cache()

    @classmethod
    def load(cls, config):
        if Llama is None:
            raise RuntimeError("未检测到 llama-cpp-python（llama_cpp）。请先安装该依赖。")

        if cls.model and cls.settings == config:
            return cls.model

        cls.unload()

        model_path = os.path.join(folder_paths.models_dir, "LLM", config["model"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件：{model_path}")

        if _bsai_check_mtp_layer(model_path):
            no_mtp_path = _bsai_strip_mtp_layer(model_path)
            if no_mtp_path and os.path.exists(no_mtp_path):
                model_path = no_mtp_path

        mmproj = config.get("mmproj", "无")
        mmproj_path = None
        if mmproj and mmproj != "无":
            mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj)
            if not os.path.exists(mmproj_path):
                raise FileNotFoundError(f"找不到 mmproj 文件：{mmproj_path}")

        family = config["family"]
        think = config.get("think", False)
        n_ctx = int(config.get("n_ctx", 8192))
        n_gpu_layers = int(config.get("n_gpu_layers", -1))

        chat_handler = None
        if mmproj_path:
            if family in ("Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL"):
                if Qwen35ChatHandler is None:
                    raise RuntimeError("当前 llama-cpp-python 不支持 Qwen35ChatHandler，请更新 llama-cpp-python。")
                try:
                    chat_handler = Qwen35ChatHandler(clip_model_path=mmproj_path, enable_thinking=think, verbose=False)
                except Exception:
                    chat_handler = Qwen35ChatHandler(clip_model_path=mmproj_path, verbose=False)
            elif family == "Qwen3-VL":
                if Qwen3VLChatHandler is None:
                    raise RuntimeError("当前 llama-cpp-python 不支持 Qwen3VLChatHandler，请更新 llama-cpp-python。")
                try:
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, force_reasoning=think, verbose=False)
                except Exception:
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, verbose=False)
            elif family == "Gemma4":
                if Gemma4ChatHandler is None:
                    raise RuntimeError("当前 llama-cpp-python 不支持 Gemma4ChatHandler，请更新 llama-cpp-python到0.3.36+。")
                try:
                    chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, enable_thinking=think, verbose=False)
                except Exception:
                    chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False)

        llama_kwargs = {
            "model_path": model_path,
            "chat_handler": chat_handler,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }

        try:
            cls.model = Llama(**llama_kwargs)
            cls.settings = dict(config)
            return cls.model
        except ValueError as e:
            if "Failed to load model from file" in str(e):
                mtp_detected = _bsai_check_mtp_layer(model_path)
                if mtp_detected:
                    no_mtp_path = _bsai_strip_mtp_layer(model_path)
                    if no_mtp_path:
                        llama_kwargs["model_path"] = no_mtp_path
                        try:
                            cls.model = Llama(**llama_kwargs)
                            cls.settings = dict(config)
                            cls.settings["model"] = os.path.basename(no_mtp_path)
                            return cls.model
                        except Exception:
                            pass
                    raise RuntimeError(
                        "模型加载失败：该 GGUF 文件包含 MTP/NextN 预测层（nextn_predict_layers），\n"
                        "当前 llama-cpp-python 版本不支持此特性。\n\n"
                        "解决方案：\n"
                        "1. 已尝试自动生成去 MTP 版本，请检查同目录下是否有 -noMTP.gguf 文件\n"
                        "2. 手动使用去 MTP 版本的 GGUF 文件\n"
                        "3. 或使用不含 MTP 层的 Qwen3.5/3.6 模型作为替代\n"
                        f"原始错误：{e}"
                    )
                raise RuntimeError(
                    "模型加载失败：Failed to load model from file\n" +
                    "可能的原因：\n" +
                    "1. 模型文件损坏或格式不兼容\n" +
                    "2. llama-cpp-python 版本不支持该模型架构\n" +
                    "3. 模型文件路径错误\n" +
                    "建议：\n" +
                    "- 检查模型文件完整性\n" +
                    "- 更新 llama-cpp-python 到最新版本\n" +
                    "- 确保模型路径正确"
                )
            if "Failed to create context with model" in str(e):
                raise RuntimeError(
                    "模型加载失败：Failed to create context with model\n" +
                    "可能的原因：\n" +
                    "1. 模型文件损坏或格式不兼容\n" +
                    "2. llama-cpp-python 版本不支持该模型\n" +
                    "3. 显存不足\n" +
                    "4. 模型文件路径错误\n" +
                    "建议：\n" +
                    "- 检查模型文件完整性\n" +
                    "- 更新 llama-cpp-python 到最新版本\n" +
                    "- 减少 GPU 层数或使用更小的模型\n" +
                    "- 确保模型路径正确"
                )
            raise


def _bsai_list_llm_files():
    folder_name = "LLM"
    llm_dir = os.path.join(folder_paths.models_dir, folder_name)
    try:
        if folder_name not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[folder_name] = ([llm_dir], {".gguf", ".safetensors", ".bin", ".pth", ".pt"})
    except Exception:
        pass
    try:
        return folder_paths.get_filename_list("LLM")
    except Exception:
        return []


def _bsai_check_mtp_layer(model_path):
    """Check if a GGUF file contains MTP/NextN prediction layer metadata."""
    import struct as _struct
    try:
        with open(model_path, 'rb') as f:
            magic = _struct.unpack('<I', f.read(4))[0]
            if magic != 0x46554747:
                return False
            version = _struct.unpack('<I', f.read(4))[0]
            tensor_count = _struct.unpack('<Q', f.read(8))[0]
            kv_count = _struct.unpack('<Q', f.read(8))[0]

            for i in range(kv_count):
                key_len = _struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8', errors='replace')
                vtype = _struct.unpack('<I', f.read(4))[0]

                if 'nextn_predict' in key:
                    return True

                if vtype == 8:
                    str_len = _struct.unpack('<Q', f.read(8))[0]
                    f.read(str_len)
                elif vtype in (4, 5, 6):
                    f.read(4)
                elif vtype == 10:
                    f.read(8)
                elif vtype == 7:
                    f.read(1)
                elif vtype == 2:
                    f.read(1)
                elif vtype == 9:
                    array_type = _struct.unpack('<I', f.read(4))[0]
                    array_len = _struct.unpack('<Q', f.read(8))[0]
                    if array_type == 8:
                        for _ in range(array_len):
                            sl = _struct.unpack('<Q', f.read(8))[0]
                            f.read(sl)
                    elif array_type in (4, 5, 6):
                        f.read(4 * array_len)
                    elif array_type == 10:
                        f.read(8 * array_len)
                    elif array_type == 7:
                        f.read(array_len)
                    elif array_type == 2:
                        f.read(array_len)
                    else:
                        break
                else:
                    break
    except Exception:
        pass
    return False


def _bsai_strip_mtp_layer(model_path):
    """Strip MTP/NextN layer from GGUF file. Returns path to stripped file or None."""
    import struct as _struct
    try:
        from gguf.constants import GGML_QUANT_SIZES
    except Exception:
        return None

    base, ext = os.path.splitext(model_path)
    no_mtp_path = base + "-noMTP" + ext
    if os.path.exists(no_mtp_path):
        return no_mtp_path

    GGUF_MAGIC = 0x46554747
    ALIGNMENT = 32
    TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    try:
        with open(model_path, 'rb') as fin:
            magic = _struct.unpack('<I', fin.read(4))[0]
            version = _struct.unpack('<I', fin.read(4))[0]
            tensor_count = _struct.unpack('<Q', fin.read(8))[0]
            kv_count = _struct.unpack('<Q', fin.read(8))[0]

            metadata = []
            for i in range(kv_count):
                key_len = _struct.unpack('<Q', fin.read(8))[0]
                key = fin.read(key_len)
                vtype = _struct.unpack('<I', fin.read(4))[0]
                pos_before = fin.tell()
                _value = None
                if vtype == 8:
                    str_len = _struct.unpack('<Q', fin.read(8))[0]
                    fin.read(str_len)
                elif vtype in TYPE_SIZES:
                    fin.read(TYPE_SIZES[vtype])
                elif vtype == 9:
                    array_type = _struct.unpack('<I', fin.read(4))[0]
                    array_len = _struct.unpack('<Q', fin.read(8))[0]
                    if array_type == 8:
                        for _ in range(array_len):
                            sl = _struct.unpack('<Q', fin.read(8))[0]
                            fin.read(sl)
                    elif array_type in (4, 5, 6):
                        fin.read(4 * array_len)
                    elif array_type == 10:
                        fin.read(8 * array_len)
                    elif array_type == 7:
                        fin.read(array_len)
                    elif array_type == 2:
                        fin.read(array_len)
                pos_after = fin.tell()
                fin.seek(pos_before)
                raw_bytes = fin.read(pos_after - pos_before)
                metadata.append((key, vtype, raw_bytes))

            tensor_infos = []
            for i in range(tensor_count):
                name_len = _struct.unpack('<Q', fin.read(8))[0]
                name = fin.read(name_len)
                n_dims = _struct.unpack('<I', fin.read(4))[0]
                dims = [_struct.unpack('<Q', fin.read(8))[0] for _ in range(n_dims)]
                ttype = _struct.unpack('<I', fin.read(4))[0]
                offset = _struct.unpack('<Q', fin.read(8))[0]
                tensor_infos.append((name, n_dims, dims, ttype, offset))

            data_start = fin.tell()
            padded_data_start = (data_start + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

        new_metadata = []
        block_count_raw = None
        for key, vtype, raw_bytes in metadata:
            key_str = key.decode('utf-8', errors='replace')
            if 'nextn_predict' in key_str:
                continue
            if key_str.endswith('.block_count'):
                old_val = _struct.unpack('<I', raw_bytes[-4:])[0]
                new_raw = raw_bytes[:-4] + _struct.pack('<I', old_val - 1)
                new_metadata.append((key, vtype, new_raw))
            else:
                new_metadata.append((key, vtype, raw_bytes))

        new_tensor_infos = [t for t in tensor_infos if b'blk.64.' not in t[0]]

        with open(no_mtp_path, 'wb') as fout:
            fout.write(_struct.pack('<I', GGUF_MAGIC))
            fout.write(_struct.pack('<I', version))
            fout.write(_struct.pack('<Q', len(new_tensor_infos)))
            fout.write(_struct.pack('<Q', len(new_metadata)))

            for key, vtype, raw_bytes in new_metadata:
                fout.write(_struct.pack('<Q', len(key)))
                fout.write(key)
                fout.write(_struct.pack('<I', vtype))
                fout.write(raw_bytes)

            current_offset = 0
            offset_map = []
            for name, n_dims, dims, ttype, old_offset in new_tensor_infos:
                fout.write(_struct.pack('<Q', len(name)))
                fout.write(name)
                fout.write(_struct.pack('<I', n_dims))
                for d in dims:
                    fout.write(_struct.pack('<Q', d))
                fout.write(_struct.pack('<I', ttype))
                fout.write(_struct.pack('<Q', current_offset))

                if ttype in GGML_QUANT_SIZES:
                    block_size, type_size = GGML_QUANT_SIZES[ttype]
                    num_elems = 1
                    for d in dims:
                        num_elems *= d
                    tensor_size = (num_elems // block_size) * type_size
                else:
                    tensor_size = 0
                offset_map.append((old_offset, current_offset, tensor_size))
                current_offset += tensor_size
                current_offset = (current_offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

            header_end = fout.tell()
            padded = (header_end + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            fout.write(b'\x00' * (padded - header_end))

            with open(model_path, 'rb') as fin:
                for old_offset, new_offset, tensor_size in offset_map:
                    fin.seek(padded_data_start + old_offset)
                    data = fin.read(tensor_size)
                    fout.write(data)
                    padded_written = (len(data) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
                    if padded_written > len(data):
                        fout.write(b'\x00' * (padded_written - len(data)))

        return no_mtp_path
    except Exception:
        return None


def _bsai_call_chat_completion(llm, messages, params):
    kwargs = dict(params or {})
    kwargs["messages"] = messages
    try:
        sig = inspect.signature(llm.create_chat_completion)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        sig = None
        has_var_kw = True

    if sig is not None and not has_var_kw:
        allowed = sig.parameters
        if "presence_penalty" in kwargs and "presence_penalty" not in allowed and "present_penalty" in allowed:
            kwargs["present_penalty"] = kwargs.pop("presence_penalty")
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return llm.create_chat_completion(**kwargs)


def _bsai_normalize_seed(seed_value):
    try:
        seed_value = int(seed_value)
    except Exception:
        return None
    if seed_value < 0:
        return None
    return seed_value


def _bsai_reset_llm_state(llm):
    try:
        ctx = getattr(llm, "_ctx", None)
        if ctx is not None and hasattr(ctx, "memory_clear"):
            ctx.memory_clear(True)
    except Exception:
        pass
    try:
        reset = getattr(llm, "reset", None)
        if callable(reset):
            reset()
        elif hasattr(llm, "n_tokens"):
            llm.n_tokens = 0
    except Exception:
        pass


class BSAI_QwenModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        all_files = _bsai_list_llm_files()
        model_list = [f for f in all_files if "mmproj" not in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin", ".pth", ".pt"]]
        mmproj_list = ["无"] + [f for f in all_files if "mmproj" in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin"]]

        if not model_list:
            model_list = ["（请把模型放到 models/LLM）"]

        return {
            "required": {
                "模型系列": (["Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL", "Gemma4"], {"default": "Qwen3.8-VL"}),
                "主模型": (model_list, {"tooltip": "主模型文件（建议 .gguf）放到 ComfyUI/models/LLM/"}),
                "视觉投影mmproj": (mmproj_list, {"default": "无", "tooltip": "多模态需要 mmproj；纯文本可选“无”。"}),
                "启用思考": ("BOOLEAN", {"default": False}),
                "上下文长度": ("INT", {"default": 16384, "min": 1024, "max": 327680, "step": 256}),
                "GPU层数": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
            }
        }

    RETURN_TYPES = ("BSAI_QWEN_MODEL",)
    RETURN_NAMES = ("qwen模型",)
    FUNCTION = "load"
    CATEGORY = "BSAI"

    def load(self, 模型系列, 主模型, 视觉投影mmproj, 启用思考, 上下文长度, GPU层数):
        if 主模型.startswith("（请把模型放到"):
            raise RuntimeError("未找到可用模型文件。请把模型放到 ComfyUI/models/LLM/ 后重启。")

        if 模型系列 in ("Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL", "Gemma4"):
            if 视觉投影mmproj == "无":
                raise RuntimeError(
                    f"{模型系列} 是多模态模型，需要选择视觉投影mmproj文件。\n" +
                    "请在 '视觉投影mmproj' 选项中选择对应的 mmproj 文件。"
                )

        config = {
            "family": 模型系列,
            "model": 主模型,
            "mmproj": 视觉投影mmproj,
            "think": bool(启用思考),
            "n_ctx": int(上下文长度),
            "n_gpu_layers": int(GPU层数),
        }
        model = _BSAI_QwenStorage.load(config)
        return (model,)


class BSAI_QwenPromptInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen模型": ("BSAI_QWEN_MODEL",),
                "输入文本": ("STRING", {"default": "", "multiline": True, "tooltip": "输入文本内容，用于提示词反推"}),
                "系统提示词": ("STRING", {"default": "请分析输入内容并生成详细描述。", "multiline": True}),
                "最大生成token": ("INT", {"default": 16384, "min": 20, "max": 65536, "step": 1}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "频率惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "存在惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("输出文本",)
    FUNCTION = "run"
    CATEGORY = "BSAI"

    def run(self, qwen模型, 输入文本, 系统提示词, 最大生成token, 温度, top_p, top_k, 重复惩罚, 频率惩罚, 存在惩罚, seed):
        llm = qwen模型

        if not hasattr(llm, 'create_chat_completion'):
            raise TypeError(f"无效的模型输入：期望 Llama 模型对象，但收到 {type(llm).__name__} 类型。请检查工作流连接，确保 'qwen模型' 输入连接到正确的模型加载器输出。")

        messages = []
        system_text = (系统提示词 or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})

        prompt_text = (输入文本 or "").strip()
        if not prompt_text:
            raise ValueError("输入文本不能为空。")

        messages.append({"role": "user", "content": prompt_text})

        try:
            max_tokens_val = int(最大生成token)
        except (TypeError, ValueError):
            raise TypeError(f"最大生成token 必须是整数类型，但收到 {type(最大生成token).__name__} 类型。请检查工作流连接。")
        
        try:
            top_k_val = int(top_k)
        except (TypeError, ValueError):
            raise TypeError(f"top_k 必须是整数类型，但收到 {type(top_k).__name__} 类型。请检查工作流连接。")

        params = {
            "max_tokens": max_tokens_val,
            "temperature": float(温度),
            "top_p": float(top_p),
            "top_k": top_k_val,
            "repeat_penalty": float(重复惩罚),
            "frequency_penalty": float(频率惩罚),
            "presence_penalty": float(存在惩罚),
            "seed": _bsai_normalize_seed(seed),
            "stream": False,
            "stop": ["</s>"],
        }

        _bsai_reset_llm_state(llm)
        try:
            out = _bsai_call_chat_completion(llm, messages=messages, params=params)
        except RuntimeError as e:
            if "Context Shift is explicitly disabled" in str(e):
                current_n_ctx = getattr(llm, "n_ctx", "未知")
                raise RuntimeError(
                    "Context Shift 被 C++ 后端禁用（M-RoPE 模型不支持上下文滑动窗口）。\n"
                    f"当前 n_ctx = {current_n_ctx}，无法容纳完整对话。\n"
                    "请在 BSAI_QwenModelLoader 节点中增大「上下文长度」：\n"
                    "  - 纯文本建议 16384\n"
                    "  - 含图片/视频建议 32768 或更高\n"
                    f"原始错误：{e}"
                ) from e
            raise

        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = str(out)

        return (text.lstrip().removeprefix(": ").strip(),)


def _bsai_get_temp_dir():
    temp_dir = Path(folder_paths.get_temp_directory()) / "bsai_multimodal"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _bsai_cleanup_old_temp_files(temp_dir, max_age_seconds=3600):
    try:
        now = time.time()
        for f in temp_dir.iterdir():
            if f.is_file() and now - f.stat().st_mtime > max_age_seconds:
                f.unlink()
    except Exception:
        pass


def _bsai_save_tensor_as_image(img_tensor, save_path, max_size=1536):
    if img_tensor is None:
        return None
    if hasattr(img_tensor, 'cpu'):
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
    else:
        img_np = img_tensor
    if img_np.shape[-1] == 3:
        img = Image.fromarray(img_np)
    elif img_np.shape[-1] == 4:
        img = Image.fromarray(img_np[:, :, :3])
    else:
        img = Image.fromarray(img_np)
    # 限制图像最大边长，防止 CLIP 编码器因分辨率过大而失败
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    img.save(save_path, quality=95)
    return str(save_path)


def _bsai_extract_video_frames(video_path, max_frames=8):
    if not video_path or not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    count = min(max_frames, total_frames)
    if count == 1:
        indices = [0]
    else:
        indices = [int(i * (total_frames - 1) / (count - 1)) for i in range(count)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


def _bsai_get_video_path(video_input):
    if video_input is None:
        return None
    if hasattr(video_input, 'get_stream_source'):
        return video_input.get_stream_source()
    if isinstance(video_input, str):
        return video_input
    return None


class BSAI_QwenMultimodalInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen模型": ("BSAI_QWEN_MODEL",),
                "输入文本": ("STRING", {"default": "", "multiline": True, "tooltip": "输入文本内容，用于提示词反推"}),
                "系统提示词": ("STRING", {"default": "请分析输入内容并生成详细描述。", "multiline": True}),
                "最大生成token": ("INT", {"default": 16384, "min": 20, "max": 65536, "step": 1}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "频率惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "存在惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "视频最大采样帧数": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "video_1": (IO.VIDEO,),
                "video_2": (IO.VIDEO,),
                "video_3": (IO.VIDEO,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("输出文本",)
    FUNCTION = "run"
    CATEGORY = "BSAI"

    def run(self, qwen模型, 输入文本, 系统提示词, 最大生成token, 温度, top_p, top_k, 重复惩罚, 频率惩罚, 存在惩罚, seed, 视频最大采样帧数, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, video_1=None, video_2=None, video_3=None):
        llm = qwen模型

        if not hasattr(llm, 'create_chat_completion'):
            raise TypeError(f"无效的模型输入：期望 Llama 模型对象，但收到 {type(llm).__name__} 类型。请检查工作流连接，确保 'qwen模型' 输入连接到正确的模型加载器输出。")

        temp_dir = _bsai_get_temp_dir()
        _bsai_cleanup_old_temp_files(temp_dir)

        content_items = []

        # 处理图像输入
        for idx, img_tensor in enumerate([image_1, image_2, image_3, image_4, image_5], 1):
            if img_tensor is not None:
                temp_path = temp_dir / f"img_{int(time.time()*1000)}_{idx}.jpg"
                saved = _bsai_save_tensor_as_image(img_tensor, temp_path)
                if saved:
                    content_items.append({"type": "image_url", "image_url": {"url": Path(saved).as_uri()}})

        # 处理视频输入
        for idx, video_input in enumerate([video_1, video_2, video_3], 1):
            video_path = _bsai_get_video_path(video_input)
            if video_path:
                frames = _bsai_extract_video_frames(video_path, max_frames=int(视频最大采样帧数))
                for fidx, frame in enumerate(frames):
                    temp_path = temp_dir / f"vid_{int(time.time()*1000)}_{idx}_f{fidx}.jpg"
                    img = Image.fromarray(frame)
                    img.save(temp_path, quality=95)
                    content_items.append({"type": "image_url", "image_url": {"url": Path(str(temp_path)).as_uri()}})

        # 文本提示
        prompt_text = (输入文本 or "").strip()
        if not prompt_text and not content_items:
            raise ValueError("输入文本和所有图像/视频不能同时为空。")
        if prompt_text:
            content_items.append({"type": "text", "text": prompt_text})

        messages = []
        system_text = (系统提示词 or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})

        messages.append({"role": "user", "content": content_items})

        try:
            max_tokens_val = int(最大生成token)
        except (TypeError, ValueError):
            raise TypeError(f"最大生成token 必须是整数类型，但收到 {type(最大生成token).__name__} 类型。请检查工作流连接。")

        try:
            top_k_val = int(top_k)
        except (TypeError, ValueError):
            raise TypeError(f"top_k 必须是整数类型，但收到 {type(top_k).__name__} 类型。请检查工作流连接。")

        params = {
            "max_tokens": max_tokens_val,
            "temperature": float(温度),
            "top_p": float(top_p),
            "top_k": top_k_val,
            "repeat_penalty": float(重复惩罚),
            "frequency_penalty": float(频率惩罚),
            "presence_penalty": float(存在惩罚),
            "seed": _bsai_normalize_seed(seed),
            "stream": False,
            "stop": ["</s>"],
        }

        _bsai_reset_llm_state(llm)
        try:
            out = _bsai_call_chat_completion(llm, messages=messages, params=params)
        except RuntimeError as e:
            if "Context Shift is explicitly disabled" in str(e):
                current_n_ctx = getattr(llm, "n_ctx", "未知")
                raise RuntimeError(
                    "Context Shift 被 C++ 后端禁用（M-RoPE 模型不支持上下文滑动窗口）。\n"
                    f"当前 n_ctx = {current_n_ctx}，无法容纳完整对话。\n"
                    "请在 BSAI_QwenModelLoader 节点中增大「上下文长度」：\n"
                    "  - 纯文本建议 16384\n"
                    "  - 含图片/视频建议 32768 或更高\n"
                    f"原始错误：{e}"
                ) from e
            raise
        except ValueError as e:
            if "Media evaluation failed" in str(e):
                raise RuntimeError(
                    "多模态图像编码失败（Media evaluation failed）。\n"
                    "可能的原因和解决方案：\n"
                    "1. 输入图像分辨率过大 — 已自动限制为最长边 1536 像素，若仍失败请手动缩小原图。\n"
                    "2. mmproj 视觉投影模型与主模型不匹配 — 请确保加载了对应版本的 mmproj 文件。\n"
                    "   例如：Qwen3.8-VL 主模型必须搭配 Qwen3.8-VL 的 mmproj。\n"
                    "3. 显存不足导致 CLIP 编码失败 — 请尝试减少输入图像/视频帧数，或重启 ComfyUI。\n"
                    "4. 图像文件损坏或格式异常 — 请检查输入图像能否正常打开。\n"
                    f"原始错误：{e}"
                ) from e
            raise

        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = str(out)

        return (text.lstrip().removeprefix(": ").strip(),)


class BSAI_QwenUnloadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"任意输入": ("*",)}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("任意输出",)
    FUNCTION = "run"
    CATEGORY = "BSAI"

    def run(self, 任意输入):
        _BSAI_QwenStorage.unload()
        return (任意输入,)


NODE_CLASS_MAPPINGS = {
    "BSAI_MultiplePathsInputPlus": BSAI_MultiplePathsInputPlus,
    "BSAI_VideoLoaderPlus": BSAI_VideoLoaderPlus,
    "MultiplePathsInputPlus": BSAI_MultiplePathsInputPlus,
    "VideoLoaderPlus": BSAI_VideoLoaderPlus,
    "BSAI_QwenModelLoader": BSAI_QwenModelLoader,
    "BSAI_QwenPromptInference": BSAI_QwenPromptInference,
    "BSAI_QwenMultimodalInference": BSAI_QwenMultimodalInference,
    "BSAI_QwenUnloadModel": BSAI_QwenUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MultiplePathsInputPlus": "BSAI Multiple Paths Input Plus",
    "BSAI_VideoLoaderPlus": "BSAI Video Loader Plus",
    "MultiplePathsInputPlus": "Multiple Paths Input Plus",
    "VideoLoaderPlus": "Video Loader Plus",
    "BSAI_QwenModelLoader": "BSAI Qwen Model Loader",
    "BSAI_QwenPromptInference": "BSAI Qwen Prompt Inference",
    "BSAI_QwenMultimodalInference": "BSAI Qwen Multimodal Inference",
    "BSAI_QwenUnloadModel": "BSAI Qwen Unload Model",
}
