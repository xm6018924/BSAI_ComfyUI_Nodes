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


def _bsai_confine_comfy_path(path):
    """Return the realpath of path if it lies inside a ComfyUI input/output/temp dir, else None."""
    try:
        real = os.path.realpath(str(path or "").strip())
    except Exception:
        return None
    if not real:
        return None
    for base in (folder_paths.get_input_directory(),
                 folder_paths.get_temp_directory(),
                 folder_paths.get_output_directory()):
        try:
            base = os.path.realpath(base)
            if os.path.commonpath([base, real]) == base:
                return real
        except Exception:
            continue
    return None


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

            # Confine user-supplied paths to ComfyUI's input/output/temp folders.
            path = _bsai_confine_comfy_path(path)
            if path is None:
                path_list.append(None)
                continue
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
                "output_mode": (["full_video", "selected_frame"], {"default": "full_video"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }

    CATEGORY = "BSAI"

    RETURN_TYPES = (IO.VIDEO, IO.VIDEO, "IMAGE", "PATH", "INT", "INT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("video", "single_frame", "single_frame_image", "path", "total_frames", "current_frame", "frames (H3 clip)", "audio (H3 clip)")
    FUNCTION = "load_video_plus"

    def _read_frame_at(self, video_path, target_frame):
        """精确读取指定帧号的帧。
        策略：先跳到目标帧前的关键帧附近，再逐帧 grab 到目标帧，确保帧号 100% 准确。
        返回 (frame_bgr, actual_frame_index)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = max(0, min(int(target_frame), total - 1))

        # Seek close to the target frame (a bit earlier so we land before a keyframe)
        seek_target = max(0, target - 60)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)

        # Actual position after set() (may be imprecise)
        actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Grab frames one by one until the target
        grabbed = 0
        while actual_pos < target:
            if not cap.grab():
                break
            actual_pos += 1
            grabbed += 1

        # Read the target frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            # Fallback: read the first frame with a fresh capture
            cap2 = cv2.VideoCapture(video_path)
            ret2, frame2 = cap2.read()
            cap2.release()
            if not ret2:
                raise ValueError(f"Cannot read video frame (target frame {target})")
            return frame2, 0

        return frame, target

    def _get_silent_audio(self, video_path, frame_rate, duration_frames=1):
        """从源视频读取音频信息，生成对应时长的静音音频。
        如果源视频没有音频，返回 None。
        """
        import torch
        try:
            import av
            try:
                container = av.open(video_path)
            except Exception:
                return None

            audio_stream = None
            for stream in container.streams:
                if stream.type == "audio":
                    audio_stream = stream
                    break

            if audio_stream is None:
                container.close()
                return None

            sample_rate = audio_stream.sample_rate
            channels = audio_stream.channels
            layout_name = audio_stream.layout.name if audio_stream.layout else ("stereo" if channels >= 2 else "mono")
            container.close()

            # 生成静音音频
            duration_sec = duration_frames / frame_rate
            num_samples = int(sample_rate * duration_sec)
            if num_samples < 1:
                num_samples = 1

            waveform = torch.zeros(1, channels, num_samples, dtype=torch.float32)

            from comfy_api.latest import Types
            return Types.AudioComponents(
                waveform=waveform,
                sample_rate=sample_rate,
            )
        except Exception:
            return None

    def load_video_plus(self, file, frame_rate=30.0, start_frame=0, end_frame=-1, loop=False,
                        resize_width=0, resize_height=0, output_mode="full_video", frame_index=0):
        video_path = folder_paths.get_annotated_filepath(file)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            actual_fps = frame_rate
        cap.release()
        
        if end_frame < 0 or end_frame > total_frames:
            end_frame = total_frames
        start_frame = max(0, min(start_frame, total_frames - 1))

        from comfy_api.latest import Types

        # ============================================================
        # 1号端口：video —— 永远输出完整视频（不受 start/end 影响）
        # ============================================================
        main_video = InputImpl.VideoFromFile(video_path)

        # ============================================================
        # 2号端口：single_frame —— 输出选中的单帧视频
        # ============================================================
        frame_bgr_single, single_idx = self._read_frame_at(video_path, frame_index)
        frame_rgb_single = cv2.cvtColor(frame_bgr_single, cv2.COLOR_BGR2RGB)
        if resize_width > 0 and resize_height > 0:
            frame_rgb_single = cv2.resize(frame_rgb_single, (resize_width, resize_height))
        frame_tensor_single = torch.from_numpy(frame_rgb_single).float() / 255.0
        frame_tensor_single = frame_tensor_single.unsqueeze(0)

        silent_audio_single = self._get_silent_audio(video_path, frame_rate, duration_frames=1)
        single_frame_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=frame_tensor_single,
                frame_rate=frame_rate,
                audio=silent_audio_single,
            ),
            bit_depth=8,
            color_space="sRGB",
        )

        # ============================================================
        # 3号端口：single_frame_image —— 单帧图片
        # ============================================================
        single_image = frame_tensor_single.clone()

        # ============================================================
        # 7号端口：frames (H3视频帧) —— 只输出 start~end 区间的帧序列
        # ============================================================
        def read_clip_frames():
            """读取 start_frame 到 end_frame（含两端）的帧序列。
            使用精确定位策略：先 seek 到起始帧附近，再逐帧读取，确保帧号 100% 准确。
            """
            cap2 = cv2.VideoCapture(video_path)
            if not cap2.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            s = max(0, min(int(start_frame), total - 1))
            e = max(0, min(int(end_frame), total - 1))
            if e < s:
                e = s

            # 先粗跳到起始帧之前（往前多跳一些，确保落在关键帧之前）
            seek_target = max(0, s - 60)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
            actual_pos = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))

            # 逐帧 grab 到起始帧位置
            while actual_pos < s:
                if not cap2.grab():
                    break
                actual_pos += 1

            # 从起始帧读到结束帧（包含两端）
            frames_list = []
            current = actual_pos
            while current <= e:
                ret, frame = cap2.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if resize_width > 0 and resize_height > 0:
                    frame_rgb = cv2.resize(frame_rgb, (resize_width, resize_height))
                frames_list.append(frame_rgb)
                current += 1

            cap2.release()

            if len(frames_list) == 0:
                # 兜底：至少返回一帧
                fallback, _ = self._read_frame_at(video_path, s)
                fallback_rgb = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                if resize_width > 0 and resize_height > 0:
                    fallback_rgb = cv2.resize(fallback_rgb, (resize_width, resize_height))
                frames_list.append(fallback_rgb)

            frames_np = np.stack(frames_list, axis=0)
            return torch.from_numpy(frames_np).float() / 255.0

        frames_out = read_clip_frames()

        # ============================================================
        # 8号端口：audio —— 只输出 start~end 区间的音频（含两端，与视频帧数对应）
        # ============================================================
        audio_out = None
        try:
            import av as _av
            container = _av.open(video_path)
            audio_stream = None
            for s in container.streams.audio:
                if s.codec_context is not None:
                    audio_stream = s
                    break
            if audio_stream is not None:
                resampler = _av.audio.resampler.AudioResampler(format='fltp', layout='stereo')
                audio_chunks = []
                for frame in container.decode(audio_stream):
                    resampled = resampler.resample(frame)
                    for rf in resampled:
                        arr = rf.to_ndarray()  # shape: (channels, samples) for fltp
                        audio_chunks.append(arr)
                container.close()
                if audio_chunks:
                    import numpy as _np
                    import torch as _torch
                    audio_data = _np.concatenate(audio_chunks, axis=-1)  # (channels, total_samples)
                    sample_rate = int(audio_stream.sample_rate)
                    waveform = _torch.from_numpy(audio_data).unsqueeze(0)  # (1, channels, total_samples)

                    total = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
                    s = max(0, min(int(start_frame), total - 1))
                    e = max(0, min(int(end_frame), total - 1))
                    if e < s:
                        e = s

                    start_sec = s / actual_fps
                    end_sec = (e + 1) / actual_fps
                    start_sample = int(start_sec * sample_rate)
                    end_sample = int(end_sec * sample_rate)

                    if start_sample < waveform.shape[-1]:
                        end_sample = min(end_sample, waveform.shape[-1])
                        clipped = waveform[:, :, start_sample:end_sample]
                        audio_out = {"waveform": clipped, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[BSAI VideoLoaderPlus] Audio extraction failed: {e}")
            audio_out = None

        return (main_video, single_frame_video, single_image, video_path, total_frames, single_idx, frames_out, audio_out)

    @classmethod
    def IS_CHANGED(cls, file, output_mode="full_video", frame_index=0, start_frame=0, end_frame=-1, **kwargs):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        return f"{mod_time}_{output_mode}_{frame_index}_{start_frame}_{end_frame}"

    @classmethod
    def VALIDATE_INPUTS(cls, file, **kwargs):
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
            raise RuntimeError("llama-cpp-python (llama_cpp) is not installed. Please install it first.")

        if cls.model and cls.settings == config:
            return cls.model

        cls.unload()

        model_path = os.path.join(folder_paths.models_dir, "LLM", config["model"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if _bsai_check_mtp_layer(model_path):
            no_mtp_path = _bsai_strip_mtp_layer(model_path)
            if no_mtp_path and os.path.exists(no_mtp_path):
                model_path = no_mtp_path

        mmproj = config.get("mmproj", "None")
        mmproj_path = None
        # "无" 为旧版哨兵标签：兼容老工作流 JSON 保存值
        if mmproj and mmproj not in ("None", "无"):
            mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj)
            if not os.path.exists(mmproj_path):
                raise FileNotFoundError(f"mmproj file not found: {mmproj_path}")

        family = config["family"]
        think = config.get("think", False)
        n_ctx = int(config.get("n_ctx", 8192))
        n_gpu_layers = int(config.get("n_gpu_layers", -1))

        chat_handler = None
        if mmproj_path:
            if family in ("Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL"):
                if Qwen35ChatHandler is None:
                    raise RuntimeError("Current llama-cpp-python does not support Qwen35ChatHandler; please update llama-cpp-python.")
                try:
                    chat_handler = Qwen35ChatHandler(clip_model_path=mmproj_path, enable_thinking=think, verbose=False, use_gpu=False, image_min_tokens=1024)
                except Exception:
                    chat_handler = Qwen35ChatHandler(clip_model_path=mmproj_path, verbose=False, use_gpu=False, image_min_tokens=1024)
            elif family == "Qwen3-VL":
                if Qwen3VLChatHandler is None:
                    raise RuntimeError("Current llama-cpp-python does not support Qwen3VLChatHandler; please update llama-cpp-python.")
                try:
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, force_reasoning=think, verbose=False, use_gpu=False, image_min_tokens=1024)
                except Exception:
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, verbose=False, use_gpu=False, image_min_tokens=1024)
            elif family == "Gemma4":
                if Gemma4ChatHandler is None:
                    raise RuntimeError("Current llama-cpp-python does not support Gemma4ChatHandler; please update llama-cpp-python to >=0.3.36.")
                try:
                    chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, enable_thinking=think, verbose=False, use_gpu=False, image_min_tokens=1024)
                except Exception:
                    chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False, use_gpu=False, image_min_tokens=1024)

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
                        "Model load failed: this GGUF file contains an MTP/NextN prediction layer "
                        "(nextn_predict_layers), which the current llama-cpp-python build does not support.\n\n"
                        "Solutions:\n"
                        "1. An attempt was made to auto-generate an MTP-stripped version; check whether a -noMTP.gguf file exists next to the original.\n"
                        "2. Manually use an MTP-stripped GGUF file.\n"
                        "3. Or use a Qwen3.5/3.6 model without the MTP layer.\n"
                        f"Original error: {e}"
                    )
                raise RuntimeError(
                    "Model load failed: Failed to load model from file\n" +
                    "Possible causes:\n" +
                    "1. The model file is corrupted or the format is unsupported\n" +
                    "2. The installed llama-cpp-python version does not support this model architecture\n" +
                    "3. The model file path is wrong\n" +
                    "Suggestions:\n" +
                    "- Verify model file integrity\n" +
                    "- Update llama-cpp-python to the latest version\n" +
                    "- Make sure the model path is correct"
                )
            if "Failed to create context with model" in str(e):
                raise RuntimeError(
                    "Model load failed: Failed to create context with model\n" +
                    "Possible causes:\n" +
                    "1. The model file is corrupted or the format is unsupported\n" +
                    "2. The installed llama-cpp-python version does not support this model\n" +
                    "3. Insufficient VRAM\n" +
                    "4. The model file path is wrong\n" +
                    "Suggestions:\n" +
                    "- Verify model file integrity\n" +
                    "- Update llama-cpp-python to the latest version\n" +
                    "- Reduce GPU layers or use a smaller model\n" +
                    "- Make sure the model path is correct"
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


def _bsai_ninfer_model_list():
    """NInfer 模型下拉：models/LLM 下所有 *.ninfer 相对路径。"""
    try:
        import importlib.util as _ilu
        _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
        if not os.path.exists(_p):
            return []
        _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        return [rel for rel, _abs in _m.NInferManager.list_models()]
    except Exception:
        return []


def _bsai_nemotron_model_list():
    """Nemotron 模型下拉：models/LLM/Nemotron 下所有非 mmproj 的 *.gguf。"""
    try:
        import importlib.util as _ilu
        _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
        if not os.path.exists(_p):
            return []
        _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        return [rel for rel, _abs, kind in _m.NemotronManager.list_models() if kind == "model"]
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


# ---------------------------------------------------------------------------
# BSAI Strata HTTP backend: drop-in replacement for llama_cpp.Llama when the
# user runs Qwen3.8-Flash-Next via the Strata server at 127.0.0.1:8080.
# ---------------------------------------------------------------------------
class _BSAI_StrataHTTPBackend:
    """Mimics the subset of Llama used by BSAI_QwenPromptInference:
    .create_chat_completion(messages=..., **params) and .n_ctx."""

    DEFAULT_BASE = "http://127.0.0.1:8080/v1"
    DEFAULT_MODEL = "qwen3.8-flash-next"

    def __init__(self, base_url=None, model=None, n_ctx=32768, quant_label=None):
        self.base_url = (base_url or self.DEFAULT_BASE).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.n_ctx = n_ctx
        self.quant_label = quant_label

    def create_chat_completion(self, messages=None, **kwargs):
        import json as _json
        import socket as _sk, http.client as _hc
        from urllib.parse import urlparse as _up
        import time as _time

        max_tokens_val = kwargs.get("max_tokens", 4096)

        payload = {
            "model": self.model,
            "messages": messages or [],
            "max_tokens": max_tokens_val,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False,
            # 思考预算：给思考留一半，给回答留一半，避免思考用完所有 token
            "reasoning_budget_tokens": min(max_tokens_val // 2, 2048),
        }
        for k in ("frequency_penalty", "presence_penalty", "seed"):
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]
        data = _json.dumps(payload).encode("utf-8")
        u = _up(self.base_url)

        def _try_connect():
            s = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
            s.settimeout(600); s.setblocking(True)
            s.connect((u.hostname, u.port or 80))
            return s

        # First attempt
        try:
            s = _try_connect()
        except Exception as e:
            print(f"[BSAI] Strata connection failed, auto-starting service... ({e})")
            # Try to start Strata via the ComfyUI API
            started = False
            for port in [8188, 8191, 8000]:
                try:
                    import urllib.request as _urllib
                    url = f"http://127.0.0.1:{port}/bsai/strata/start"
                    req = _urllib.request.Request(url)
                    with _urllib.request.urlopen(req, timeout=5) as resp:
                        result = _json.loads(resp.read().decode())
                        if result.get("ok"):
                            print(f"[BSAI] Strata started successfully (pid={result.get('pid')})")
                            started = True
                            break
                except Exception:
                    continue

            if not started:
                raise ConnectionError(
                    f"Cannot connect to Strata service ({self.base_url}); auto-start also failed.\n"
                    "Please run START-HERE.bat to start the Strata service, then run the workflow."
                )

            # Wait for the service to become ready
            print("[BSAI] Waiting for Strata service to become ready...")
            for i in range(30):
                _time.sleep(2)
                try:
                    s = _try_connect()
                    print("[BSAI] Strata service is ready")
                    break
                except Exception:
                    if i == 29:
                        raise ConnectionError("Strata service startup timed out (60s); please check the logs.")
                    continue
            else:
                raise ConnectionError("Strata service startup timed out.")

        try:
            conn = _hc.HTTPConnection(u.hostname, u.port or 80, timeout=600)
            conn.sock = s
            conn.request("POST", u.path + "/chat/completions", body=data,
                         headers={"Content-Type": "application/json", "Authorization": "Bearer strata"})
            return _json.loads(conn.getresponse().read().decode("utf-8", "replace"))
        finally:
            s.close()

    def close(self):
        pass


# BSAI NInfer HTTP backend: same Llama-mimicking interface, but talks to the
# local NInfer engine (AI蛮子 plugin) whose port is managed dynamically by
# NInferManager inside BSAI-LLM-Strata/strata_manager.py.
# ---------------------------------------------------------------------------
class _BSAI_NInferHTTPBackend:
    """Mimics the subset of Llama used by BSAI_QwenPromptInference, routed to
    the local NInfer reverse engine (auto-started, dynamic port 808x)."""

    def __init__(self, n_ctx=32768, quant_label=None, ninfer_model="(Auto — first model found)"):
        self.n_ctx = n_ctx
        self.quant_label = quant_label
        self.ninfer_model = None if str(ninfer_model).startswith("(Auto") else str(ninfer_model)
        self._mgr = None

    def _manager(self):
        if self._mgr is None:
            import importlib.util as _ilu
            _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
            _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
            _m = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            self._mgr = _m.NInferManager
        return self._mgr

    def create_chat_completion(self, messages=None, **kwargs):
        mgr = self._manager()
        ok, msg = mgr.ensure_running(model_path=self.ninfer_model)
        if not ok:
            raise ConnectionError(f"NInfer engine failed to start: {msg}")
        import json as _json
        import socket as _sk, http.client as _hc
        from urllib.parse import urlparse as _up

        max_tokens_val = kwargs.get("max_tokens", 4096)
        payload = {
            "messages": messages or [],
            "max_tokens": max_tokens_val,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False,
        }
        mids = mgr.models()
        payload["model"] = mids[0] if mids else "default"
        data = _json.dumps(payload).encode("utf-8")
        u = _up(mgr.base_url())

        s = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
        s.settimeout(600); s.setblocking(True)
        s.connect((u.hostname, u.port or 80))
        try:
            conn = _hc.HTTPConnection(u.hostname, u.port or 80, timeout=600)
            conn.sock = s
            conn.request("POST", u.path + "/chat/completions", body=data,
                         headers={"Content-Type": "application/json"})
            return _json.loads(conn.getresponse().read().decode("utf-8", "replace"))
        finally:
            s.close()

    def close(self):
        pass


# BSAI Nemotron HTTP backend: same Llama-mimicking interface, routed to the
# local Nemotron 3.5 Bolt (llama.cpp @127.0.0.1:8085), auto-started by
# NemotronManager inside BSAI-LLM-Strata/strata_manager.py.
# ---------------------------------------------------------------------------
class _BSAI_NemotronHTTPBackend:
    """Mimics the subset of Llama used by BSAI_QwenPromptInference, routed to
    the local Nemotron 3.5 Bolt engine (llama.cpp, OpenAI-compatible @8085)."""

    def __init__(self, n_ctx=32768, quant_label=None):
        self.n_ctx = n_ctx
        self.quant_label = quant_label
        self._mgr = None

    def _manager(self):
        if self._mgr is None:
            import importlib.util as _ilu
            _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
            _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
            _m = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            self._mgr = _m.NemotronManager
        return self._mgr

    def create_chat_completion(self, messages=None, **kwargs):
        mgr = self._manager()
        ok, msg = mgr.ensure_running()
        if not ok:
            raise ConnectionError(f"Nemotron engine failed to start: {msg}")
        import json as _json
        import socket as _sk, http.client as _hc
        from urllib.parse import urlparse as _up

        max_tokens_val = kwargs.get("max_tokens", 4096)
        payload = {
            "messages": messages or [],
            "max_tokens": max_tokens_val,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False,
        }
        mids = mgr.models()
        payload["model"] = mids[0] if mids else "Nemotron-3.5-Bolt-74B-A9B"
        data = _json.dumps(payload).encode("utf-8")
        u = _up(mgr.base_url())

        s = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
        s.settimeout(600); s.setblocking(True)
        s.connect((u.hostname, u.port or 80))
        try:
            conn = _hc.HTTPConnection(u.hostname, u.port or 80, timeout=600)
            conn.sock = s
            conn.request("POST", u.path + "/chat/completions", body=data,
                         headers={"Content-Type": "application/json",
                                  "Authorization": "Bearer strata"})
            return _json.loads(conn.getresponse().read().decode("utf-8", "replace"))
        finally:
            s.close()

    def close(self):
        pass


class BSAI_QwenModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        all_files = _bsai_list_llm_files()
        model_list = [f for f in all_files if "mmproj" not in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin", ".pth", ".pt"]]
        mmproj_list = ["None"] + [f for f in all_files if "mmproj" in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin"]]

        if not model_list:
            model_list = ["(place models in models/LLM)"]

        strata_quants = ["Auto (auto)", "Q2_0 (fastest)", "IQ2_XS (slightly better)", "IQ3_XXS (better)", "IQ3_S (best)"]

        return {
            "required": {
                "backend": (["llama.cpp (local GGUF)", "BSAI Strata HTTP (Qwen3.8-Flash-Next @127.0.0.1:8080)", "BSAI Nemotron HTTP (llama.cpp @127.0.0.1:8085)", "BSAI NInfer HTTP (local reverse @127.0.0.1:808x)", "BSAI NInfer HTTP (本地反推 @127.0.0.1:808x)"],
                          {"default": "llama.cpp (local GGUF)",
                           "tooltip": "Strata backend calls the local Strata service (8080); Nemotron calls llama.cpp (8085); NInfer calls the local ninfer-serve engine on a dynamic free port (8081+)"}),
                "model_family": (["Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL", "Gemma4"], {"default": "Qwen3.8-VL"}),
                "model_file": (model_list, {"tooltip": "Main model file (recommended .gguf) placed in ComfyUI/models/LLM/"}),
                "mmproj": (mmproj_list, {"default": "None", "tooltip": "Multimodal requires mmproj; for text-only you may select None."}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "context_length": ("INT", {"default": 32768, "min": -1, "max": 327680, "step": 256,
                                            "tooltip": "上下文长度; -1 = 自动(32768), 兼容旧工作流保存的 -1"}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
                "strata_quant": (strata_quants, {"default": "Auto (auto)", "tooltip": "Strata backend only: select quantization level; Auto picks the fastest"}),
                "ninfer_model": (["(Auto — first model found)"] + _bsai_ninfer_model_list(),
                                 {"default": "(Auto — first model found)",
                                  "tooltip": "NInfer backend only: which *.ninfer model to load (models/LLM). Changing it restarts the NInfer engine."}),
                "nemotron_model": (["(Auto — first model found)"] + _bsai_nemotron_model_list(),
                                 {"default": "(Auto — first model found)",
                                  "tooltip": "Nemotron backend only: which GGUF model to load (models/LLM/Nemotron, excludes mmproj). Changing it restarts llama.cpp."}),
            }
        }

    RETURN_TYPES = ("BSAI_QWEN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "BSAI"

    def load(self, backend, model_family, model_file, mmproj, enable_thinking, context_length, gpu_layers, strata_quant="Auto (auto)", ninfer_model="(Auto — first model found)", nemotron_model="(Auto — first model found)"):
        # 兼容旧工作流: context_length=-1 (旧版 auto 语义) -> 默认 32768
        ctx = int(context_length)
        if ctx < 1024:
            ctx = 32768
        # --- Strata HTTP backend: skip llama.cpp loading entirely ---
        if str(backend).startswith("BSAI Strata"):
            # Parse quant label (含旧版中文标签映射)
            quant_map = {
                "Auto (auto)": None,
                "Auto (自动适配)": None,
                "Q2_0 (fastest)": "Q2_0",
                "Q2_0 (最快 / fastest)": "Q2_0",
                "IQ2_XS (slightly better)": "IQ2_XS",
                "IQ2_XS (略好 / better)": "IQ2_XS",
                "IQ3_XXS (better)": "IQ3_XXS",
                "IQ3_XXS (更好 / good)": "IQ3_XXS",
                "IQ3_S (best)": "IQ3_S",
                "IQ3_S (最好 / best quality)": "IQ3_S",
                # i18n 前本节点的旧版标签
                "Q2_0 (最快)": "Q2_0",
                "IQ2_XS (略好)": "IQ2_XS",
                "IQ3_XXS (更好)": "IQ3_XXS",
                "IQ3_S (最好)": "IQ3_S",
            }
            quant = quant_map.get(strata_quant, None)

            # Auto-check and start the Strata service
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                result = s.connect_ex(("127.0.0.1", 8080))
                s.close()
                if result != 0:
                    print("[BSAI] Strata service is not running, auto-starting...")
                    # Call the ComfyUI API to start Strata
                    try:
                        import urllib.request
                        import json
                        # Try common ports
                        for port in [8188, 8191, 8000]:
                            try:
                                url = f"http://127.0.0.1:{port}/bsai/strata/start"
                                req = urllib.request.Request(url)
                                with urllib.request.urlopen(req, timeout=5) as resp:
                                    data = json.loads(resp.read().decode())
                                    if data.get("ok"):
                                        print(f"[BSAI] Strata started successfully (pid={data.get('pid')})")
                                        # Wait for the service to start
                                        import time
                                        for i in range(30):
                                            time.sleep(2)
                                            s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                            s2.settimeout(2)
                                            r2 = s2.connect_ex(("127.0.0.1", 8080))
                                            s2.close()
                                            if r2 == 0:
                                                print("[BSAI] Strata service is ready")
                                                break
                                        break
                            except Exception:
                                continue
                    except Exception as e:
                        print(f"[BSAI] Auto-start Strata failed: {e}")
                        print("[BSAI] Please run START-HERE.bat to start the Strata service manually")
            except Exception:
                pass

            return (_BSAI_StrataHTTPBackend(n_ctx=ctx, quant_label=quant),)

        # --- Nemotron HTTP backend: talk to the local llama.cpp engine ---
        if str(backend).startswith("BSAI Nemotron"):
            try:
                import importlib.util as _ilu
                _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
                _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
                _m = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_m)
                _mp = None if str(nemotron_model).startswith("(Auto") else str(nemotron_model)
                ok, msg = _m.NemotronManager.ensure_running(model_path=_mp)
                if not ok:
                    raise RuntimeError(f"Nemotron engine failed to start: {msg}")
                print(f"[BSAI] Nemotron service ready @{_m.NemotronManager.base_url()}")
            except Exception as e:
                print(f"[BSAI] Auto-start Nemotron failed: {e}")
                print("[BSAI] 请确认 llama.cpp 已放入 BSAI-LLM-Strata/Nemotron/llama.cpp/，"
                      "模型已下载到 models/LLM/Nemotron/")
            return (_BSAI_NemotronHTTPBackend(n_ctx=ctx),)

        # --- NInfer HTTP backend: talk to the AI蛮子 ninfer-serve engine ---
        if str(backend).startswith("BSAI NInfer"):
            try:
                import importlib.util as _ilu
                _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
                _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
                _m = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_m)
                ok, msg = _m.NInferManager.ensure_running()
                if not ok:
                    raise RuntimeError(f"NInfer engine failed to start: {msg}")
                print(f"[BSAI] NInfer service ready @{_m.NInferManager.base_url()}")
            except Exception as e:
                print(f"[BSAI] Auto-start NInfer failed: {e}")
                print("[BSAI] Please check BSAI-FLUX3-ComfyUI/engine/ and *.ninfer model in models/LLM")
            return (_BSAI_NInferHTTPBackend(n_ctx=ctx, ninfer_model=ninfer_model),)

        if model_file is None or str(model_file).startswith("(place models") or str(model_file).startswith("（请把模型放到"):
            raise RuntimeError("No model file found. Please place the model in ComfyUI/models/LLM/ and restart.")

        if model_family in ("Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL", "Gemma4"):
            # "无" 为旧版哨兵标签：兼容老工作流 JSON 保存值
            if str(mmproj) in ("None", "无"):
                raise RuntimeError(
                    f"{model_family} is a multimodal model and requires an mmproj file.\n" +
                    "Please select the matching mmproj file in the 'mmproj' option."
                )

        config = {
            "family": model_family,
            "model": model_file,
            "mmproj": mmproj,
            "think": bool(enable_thinking),
            "n_ctx": ctx,
            "n_gpu_layers": int(gpu_layers),
        }
        model = _BSAI_QwenStorage.load(config)
        return (model,)


class BSAI_QwenPromptInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_model": ("BSAI_QWEN_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Input text content for prompt reverse-engineering"}),
                "system_prompt": ("STRING", {"default": "Analyze the input and generate a detailed description.", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 20, "max": 65536, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "auto_unload": ("BOOLEAN", {"default": True, "tooltip": "Immediately release LLM VRAM (~16GB) after inference to avoid OOM for later models. Disable when chaining multiple nodes on the same model."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "BSAI"

    def run(self, qwen_model, text, system_prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, frequency_penalty, presence_penalty, seed, auto_unload):
        llm = qwen_model

        if not hasattr(llm, 'create_chat_completion'):
            raise TypeError(f"Invalid model input: expected a Llama model object but got {type(llm).__name__}. Please check the workflow wiring and connect the 'qwen_model' input to the correct model loader output.")

        messages = []
        system_text = (system_prompt or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})

        prompt_text = (text or "").strip()
        if not prompt_text:
            raise ValueError("Input text must not be empty.")

        messages.append({"role": "user", "content": prompt_text})

        try:
            max_tokens_val = int(max_tokens)
        except (TypeError, ValueError):
            raise TypeError(f"max_tokens must be an integer, but got {type(max_tokens).__name__}. Please check the workflow wiring.")

        try:
            top_k_val = int(top_k)
        except (TypeError, ValueError):
            raise TypeError(f"top_k must be an integer, but got {type(top_k).__name__}. Please check the workflow wiring.")

        params = {
            "max_tokens": max_tokens_val,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": top_k_val,
            "repeat_penalty": float(repeat_penalty),
            "frequency_penalty": float(frequency_penalty),
            "presence_penalty": float(presence_penalty),
            "seed": _bsai_normalize_seed(seed),
            "stream": False,
            "stop": ["</s>"],
        }

        _bsai_reset_llm_state(llm)
        try:
            out = _bsai_call_chat_completion(llm, messages=messages, params=params)
        except RuntimeError as e:
            if "Context Shift is explicitly disabled" in str(e):
                current_n_ctx = getattr(llm, "n_ctx", "unknown")
                raise RuntimeError(
                    "Context Shift is explicitly disabled by the C++ backend "
                    "(M-RoPE models do not support context sliding window).\n"
                    f"Current n_ctx = {current_n_ctx}, which cannot hold the full conversation.\n"
                    "Please increase 'context_length' in the BSAI_QwenModelLoader node:\n"
                    "  - text-only: 16384 recommended\n"
                    "  - with images/video: 32768 or higher recommended\n"
                    f"Original error: {e}"
                ) from e
            raise

        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = str(out)

        # Strata HTTP may return None; fall back to an empty string
        if text is None:
            text = ""

        if auto_unload:
            _BSAI_QwenStorage.unload()

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


# ---------------------------------------------------------------------------
# Audio helpers: ComfyUI AUDIO dict -> numpy mono / base64 wav, plus offline
# Whisper large-v3 ASR across the multi-GPU package:
#   - "auto"      -> persistent subprocess worker running in python_embeded
#                    (torch+xpu) so Whisper lives on the idle Intel iGPU and
#                    never touches the 5090 that hosts the LLM.
#   - "cuda"      -> inline fp16 on the local 5090 (fast but uses ~3GB VRAM)
#   - "cpu"       -> inline fp32 on CPU (slow, zero GPU use)
# ---------------------------------------------------------------------------
_BSAI_WHISPER_CACHE = {}
_BSAI_WHISPER_WORKER = None


def _bsai_comfy_audio_to_np(audio_input):
    """ComfyUI AUDIO dict {"waveform": tensor, "sample_rate": int} -> (np.float32 mono, sr) or (None, None)."""
    try:
        if not isinstance(audio_input, dict):
            return None, None
        wf = audio_input.get("waveform")
        if wf is None:
            return None, None
        sr = int(audio_input.get("sample_rate", 16000) or 16000)
        a = wf.detach().cpu().float()
        if a.ndim == 3:
            a = a[0]  # take first item in batch
        if a.ndim == 2:
            # channels-first (C,T) vs channels-last (T,C): T is the longer dim
            if a.shape[0] >= a.shape[1]:
                a = a.mean(dim=1)
            else:
                a = a.mean(dim=0)
        a = a.numpy()
        if a.ndim != 1:
            a = a.reshape(-1)
        return a.astype(np.float32), sr
    except Exception:
        return None, None


def _bsai_asr_fmt_ts(sec):
    """秒 -> 'mm:ss' 时间戳（分段转写用）。"""
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(round(sec - m * 60))
    if s >= 60:
        m += 1
        s -= 60
    return f"{m:02d}:{s:02d}"


def _bsai_asr_lang_code(label):
    """'中文 (zh)' -> 'zh'；'自动检测'/'未知' -> None（不传 language，Whisper 自行检测）。"""
    import re as _re
    m = _re.search(r"\(([a-z]{2})\)", str(label or ""))
    return m.group(1) if m else None


# 低于该 RMS(约-49dBFS)的 30s 块视为静音, 直接标注跳过, 不进 Whisper
# (防止长前奏/间奏静音段触发 Whisper 幻觉文本, 同时省推理时间)
_BSAI_ASR_SILENCE_RMS = 0.0035


def _bsai_asr_transcribe_segments(audio_16k, proc, model, dev, language=None):
    """30s 分块转写 -> 带时间戳的分段文本。

    每块: 静音(RMS<阈值)标「（无人声）」跳过; 其余 generate(强制 language) 转写,
    空结果标「（无人声/纯音乐）」。输出格式 '[mm:ss-mm:ss] 文本' 每行一段,
    为下游 LLM 提供歌词时间轴锚点。
    """
    win = 30 * 16000
    segs = []
    gen_kw = {"task": "transcribe"}
    if language:
        gen_kw["language"] = language
    for i in range(0, len(audio_16k), win):
        chunk = audio_16k[i:i + win]
        if len(chunk) < 8000:  # skip <0.5s tail
            continue
        t0, t1 = i / 16000.0, (i + len(chunk)) / 16000.0
        ts = f"[{_bsai_asr_fmt_ts(t0)}-{_bsai_asr_fmt_ts(t1)}]"
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < _BSAI_ASR_SILENCE_RMS:
            segs.append(f"{ts}（无人声）")
            continue
        inputs = proc(chunk, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features.to(dev).to(model.dtype)
        import torch as _t
        with _t.no_grad():
            pred = model.generate(feat, **gen_kw)
        text = proc.batch_decode(pred, skip_special_tokens=True)[0].strip()
        segs.append(f"{ts} {text}" if text else f"{ts}（无人声/纯音乐）")
    return "\n".join(segs)


def _bsai_audio_to_temp_wav(audio_input, temp_dir):
    """ComfyUI AUDIO dict -> temp wav path (for the XPU worker), or None."""
    audio, sr = _bsai_comfy_audio_to_np(audio_input)
    if audio is None:
        return None
    try:
        import soundfile as sf
        p = os.path.join(str(temp_dir), f"whisper_{int(time.time()*1000)}.wav")
        sf.write(p, audio, int(sr))
        return p
    except Exception:
        return None


def _bsai_get_whisper_worker():
    """Spawn (once) the persistent XPU whisper worker in python_embeded.
    The worker warms up its XPU JIT on first launch (can take minutes), then
    stays resident and answers stdin JSON requests line by line."""
    global _BSAI_WHISPER_WORKER
    if _BSAI_WHISPER_WORKER is not None and _BSAI_WHISPER_WORKER.poll() is None:
        return _BSAI_WHISPER_WORKER
    import subprocess
    import sys as _sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.realpath(os.path.join(here, "..", "..", "..", ".."))
    xpu_py = os.path.join(repo_root, "python_embeded", "python.exe")
    if not os.path.exists(xpu_py):
        xpu_py = _sys.executable
    worker_script = os.path.join(here, "_bsai_whisper_worker.py")
    if not os.path.exists(worker_script):
        raise FileNotFoundError(f"Whisper worker script not found: {worker_script}")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(
        [xpu_py, "-X", "utf8", worker_script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", bufsize=1, env=env,
    )
    _BSAI_WHISPER_WORKER = p
    return p


def _bsai_whisper_via_worker(wav_path, language=None):
    import json as _json
    p = _bsai_get_whisper_worker()
    p.stdin.write(_json.dumps({"wav": wav_path, "language": language}) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        raise RuntimeError("Whisper worker exited unexpectedly while transcribing.")
    resp = _json.loads(line)
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "Whisper worker transcription failed."))
    return resp.get("text", "")


def _bsai_whisper_inline(audio, sr, device, language=None):
    """Inline whisper transcription inside the ComfyUI process (CUDA fp16 / CPU fp32)."""
    global _BSAI_WHISPER_CACHE
    import torch as _t
    import warnings as _w
    _w.filterwarnings("ignore")
    if "proc" not in _BSAI_WHISPER_CACHE or _BSAI_WHISPER_CACHE.get("dev") != device:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        md = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "whisper", "whisper-large-v3"))
        if not os.path.isdir(md):
            raise FileNotFoundError(f"Whisper model directory not found: {md}")
        proc = WhisperProcessor.from_pretrained(md)
        if device == "cuda" and not _t.cuda.is_available():
            device = "cpu"
        dtype = _t.float16 if device == "cuda" else _t.float32
        mdl = WhisperForConditionalGeneration.from_pretrained(md, torch_dtype=dtype).to(device).eval()
        _BSAI_WHISPER_CACHE.update(proc=proc, mdl=mdl, dev=device)
    proc, mdl, dev = _BSAI_WHISPER_CACHE["proc"], _BSAI_WHISPER_CACHE["mdl"], _BSAI_WHISPER_CACHE["dev"]
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return _bsai_asr_transcribe_segments(
        np.asarray(audio, dtype=np.float32), proc, mdl, dev, language=language)


def _bsai_whisper_transcribe(audio_input, device_pref, temp_dir, language=None):
    """Dispatch ASR by device preference:
       auto -> persistent XPU (Intel iGPU) worker; cuda -> inline fp16 on 5090; cpu -> inline CPU.
       language: ISO-639-1 code (e.g. 'zh') to force decoder language; None = auto-detect."""
    pref = str(device_pref)
    if pref.startswith("本机GPU"):
        audio, sr = _bsai_comfy_audio_to_np(audio_input)
        if audio is None:
            return ""
        return _bsai_whisper_inline(audio, sr, "cuda", language=language)
    if pref.startswith("CPU"):
        audio, sr = _bsai_comfy_audio_to_np(audio_input)
        if audio is None:
            return ""
        return _bsai_whisper_inline(audio, sr, "cpu", language=language)
    # auto: XPU worker on the idle Intel iGPU
    wav_path = _bsai_audio_to_temp_wav(audio_input, temp_dir)
    if not wav_path:
        return ""
    return _bsai_whisper_via_worker(wav_path, language=language)


def _bsai_comfy_audio_to_wav_b64(audio_input):
    """ComfyUI AUDIO dict -> base64-encoded WAV string for OpenAI input_audio content, or None."""
    audio, sr = _bsai_comfy_audio_to_np(audio_input)
    if audio is None:
        return None
    try:
        import io
        import base64 as _b64
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr), format="WAV")
        return _b64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


class BSAI_QwenMultimodalInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_model": ("BSAI_QWEN_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Input text content for prompt reverse-engineering"}),
                "system_prompt": ("STRING", {"default": "Analyze the input and generate a detailed description.", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 20, "max": 65536, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "auto_unload": ("BOOLEAN", {"default": True, "tooltip": "Immediately release LLM VRAM (~16GB) after inference to avoid OOM for later models. Disable when chaining multiple nodes on the same model."}),
                "video_max_frames": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "audio_mode": (["Local ASR transcription", "Native audio modality", "本地ASR转写（推荐：兼容所有模型，提取字幕）", "原生音频模态（input_audio：需Omni/音频模型）"], {"default": "Local ASR transcription", "tooltip": "Local ASR: transcribes audio with Whisper large-v3 into subtitle text injected into prompt (works with all models). Native audio: sends audio as input_audio directly to backend (Qwen2.5-Omni etc.)."}),
                "whisper_device": (["Auto (Intel XPU preferred)", "GPU CUDA (fastest)", "CPU (no GPU)", "自动（Intel核显XPU优先，不占5090显存）"], {"default": "Auto (Intel XPU preferred)", "tooltip": "ASR device: Auto prefers idle Intel iGPU for the resident worker; GPU runs fp16 on CUDA (fastest, ~3GB VRAM); CPU fallback."}),
                "whisper_language": (["Chinese (zh)", "Auto detect", "English (en)", "Japanese (ja)", "Korean (ko)", "中文 (zh)", "自动检测"], {"default": "Chinese (zh)", "tooltip": "ASR language: forcing zh for Chinese content reduces language-misdetection hallucination in long intros; Auto lets Whisper decide."}),
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
                "audio_1": (IO.AUDIO,),
                "audio_2": (IO.AUDIO,),
                "audio_3": (IO.AUDIO,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "BSAI"

    def run(self, qwen_model, text, system_prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, frequency_penalty, presence_penalty, seed, auto_unload, video_max_frames, audio_mode, whisper_device, whisper_language="Chinese (zh)", image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, video_1=None, video_2=None, video_3=None, audio_1=None, audio_2=None, audio_3=None):
        # Legacy Chinese dropdown values → English (backward compat for old workflows)
        _audio_map = {
            "本地ASR转写（推荐：兼容所有模型，提取字幕）": "Local ASR transcription",
            "原生音频模态（input_audio：需Omni/音频模型）": "Native audio modality",
        }
        _dev_map = {
            "自动（Intel核显XPU优先，不占5090显存）": "Auto (Intel XPU preferred)",
            "本机GPU CUDA（5090，最快但占~3GB显存）": "GPU CUDA (fastest)",
            "CPU（最慢，零GPU占用）": "CPU (no GPU)",
        }
        _lang_map = {
            "中文 (zh)": "Chinese (zh)", "自动检测": "Auto detect",
            "英语 (en)": "English (en)", "日语 (ja)": "Japanese (ja)", "韩语 (ko)": "Korean (ko)",
        }
        audio_mode = _audio_map.get(audio_mode, audio_mode)
        whisper_device = _dev_map.get(whisper_device, whisper_device)
        whisper_language = _lang_map.get(whisper_language, whisper_language)

        llm = qwen_model

        if not hasattr(llm, 'create_chat_completion'):
            raise TypeError(f"Invalid model input: expected a Llama model object but got {type(llm).__name__}. Please check the workflow wiring and connect the 'qwen_model' input to the correct model loader output.")

        temp_dir = _bsai_get_temp_dir()
        _bsai_cleanup_old_temp_files(temp_dir)

        content_items = []

        # Process image inputs
        for idx, img_tensor in enumerate([image_1, image_2, image_3, image_4, image_5], 1):
            if img_tensor is not None:
                temp_path = temp_dir / f"img_{int(time.time()*1000)}_{idx}.jpg"
                saved = _bsai_save_tensor_as_image(img_tensor, temp_path)
                if saved:
                    content_items.append({"type": "image_url", "image_url": {"url": Path(saved).as_uri()}})

        # Process video inputs
        for idx, video_input in enumerate([video_1, video_2, video_3], 1):
            video_path = _bsai_confine_comfy_path(_bsai_get_video_path(video_input))
            if video_path:
                frames = _bsai_extract_video_frames(video_path, max_frames=int(video_max_frames))
                for fidx, frame in enumerate(frames):
                    temp_path = temp_dir / f"vid_{int(time.time()*1000)}_{idx}_f{fidx}.jpg"
                    img = Image.fromarray(frame)
                    img.save(temp_path, quality=95)
                    content_items.append({"type": "image_url", "image_url": {"url": Path(str(temp_path)).as_uri()}})

        # Process audio inputs: local Whisper ASR transcription (default) or
        # raw input_audio passthrough for native audio-modal (Omni) models.
        native_audio = str(audio_mode).startswith("Native")
        for idx, audio_in in enumerate([audio_1, audio_2, audio_3], 1):
            if audio_in is None:
                continue
            if native_audio:
                b64 = _bsai_comfy_audio_to_wav_b64(audio_in)
                if b64:
                    content_items.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
            else:
                try:
                    # 客观元数据(时长/采样率)直接从 AUDIO tensor 测量, 防止 LLM 编造时长
                    _a_np, _a_sr = _bsai_comfy_audio_to_np(audio_in)
                    _dur = (len(_a_np) / float(_a_sr)) if (_a_np is not None and _a_sr) else 0.0
                    transcript = _bsai_whisper_transcribe(
                        audio_in, whisper_device, temp_dir,
                        language=_bsai_asr_lang_code(whisper_language))
                    if transcript:
                        _dur_str = f"{int(_dur // 60)}分{_dur % 60:.1f}秒" if _dur > 0 else "未知"
                        _meta = (f"[音频{idx} 元数据] 时长 {_dur_str} ({_dur:.1f}秒)"
                                 f" | 采样率 {int(_a_sr)}Hz | 本地 Whisper-large-v3 转写")
                        _rules = (
                            f"[音频{idx} ASR使用规则] 上述转写为机器自动生成，唱腔/混响可能造成错字漏句，仅供参考。"
                            f"① 涉及时长、CLIP/分镜数量的任何计算，必须以上方「时长」元数据为准，禁止自行推算或编造；"
                            f"② 歌名/歌手等身份信息只能依据转写文本中真实出现的歌词推断，且须引用转写原文佐证，"
                            f"无法确认时必须明确写「依据现有转写无法确认」，禁止凭记忆猜测或联想；"
                            f"③ 标注「（无人声）」或转写为空的段落是纯音乐/静音/间奏，不得为其虚构歌词。")
                        content_items.append({"type": "text", "text":
                                              f"{_meta}\n[音频{idx} 分段转写]\n{transcript}\n{_rules}"})
                except Exception as e:
                    import logging as _lg
                    _lg.warning(f"[BSAI] audio_{idx} ASR transcription failed: {e}")

        # Text prompt
        prompt_text = (text or "").strip()
        if not prompt_text and not content_items:
            raise ValueError("Input text and all image/video inputs cannot be empty at the same time.")
        if prompt_text:
            content_items.append({"type": "text", "text": prompt_text})

        # If we are on the Strata/NInfer HTTP backend, convert local file:// image
        # URLs to base64 data URLs (OpenAI-compatible servers cannot read local paths).
        if isinstance(llm, (_BSAI_StrataHTTPBackend, _BSAI_NInferHTTPBackend)):
            import base64 as _b64
            converted = []
            for item in content_items:
                if item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("file://"):
                        try:
                            # Fix Windows file:///C:/... format: strip leading slash after file://
                            local = url[7:]
                            if local.startswith("/"):
                                local = local[1:]
                            with open(local, "rb") as _f:
                                b = _b64.b64encode(_f.read()).decode("ascii")
                            ext = os.path.splitext(local)[1].lower().lstrip(".")
                            mime = "jpeg" if ext in ("jpg", "jpeg") else ("png" if ext == "png" else "jpeg")
                            item = {"type": "image_url",
                                    "image_url": {"url": f"data:image/{mime};base64,{b}"}}
                        except Exception as e:
                            import logging
                            logging.warning(f"[BSAI] Failed to convert image to base64: {e}, url={url}")
                            pass
                converted.append(item)
            content_items = converted

        messages = []
        system_text = (system_prompt or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})

        messages.append({"role": "user", "content": content_items})

        try:
            max_tokens_val = int(max_tokens)
        except (TypeError, ValueError):
            raise TypeError(f"max_tokens must be an integer, but got {type(max_tokens).__name__}. Please check the workflow wiring.")

        try:
            top_k_val = int(top_k)
        except (TypeError, ValueError):
            raise TypeError(f"top_k must be an integer, but got {type(top_k).__name__}. Please check the workflow wiring.")

        params = {
            "max_tokens": max_tokens_val,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": top_k_val,
            "repeat_penalty": float(repeat_penalty),
            "frequency_penalty": float(frequency_penalty),
            "presence_penalty": float(presence_penalty),
            "seed": _bsai_normalize_seed(seed),
            "stream": False,
            "stop": ["</s>"],
        }

        _bsai_reset_llm_state(llm)
        try:
            out = _bsai_call_chat_completion(llm, messages=messages, params=params)
        except RuntimeError as e:
            if "Context Shift is explicitly disabled" in str(e):
                current_n_ctx = getattr(llm, "n_ctx", "unknown")
                raise RuntimeError(
                    "Context Shift is explicitly disabled by the C++ backend "
                    "(M-RoPE models do not support context sliding window).\n"
                    f"Current n_ctx = {current_n_ctx}, which cannot hold the full conversation.\n"
                    "Please increase 'context_length' in the BSAI_QwenModelLoader node:\n"
                    "  - text-only: 16384 recommended\n"
                    "  - with images/video: 32768 or higher recommended\n"
                    f"Original error: {e}"
                ) from e
            raise
        except ValueError as e:
            if "Media evaluation failed" in str(e):
                raise RuntimeError(
                    "Multimodal image encoding failed (Media evaluation failed).\n"
                    "Possible causes and solutions:\n"
                    "1. Input image resolution is too large - it is automatically limited to a max edge of 1536 px; if it still fails, downscale the source image manually.\n"
                    "2. The mmproj vision projector does not match the main model - make sure the matching mmproj is loaded.\n"
                    "   For example: the Qwen3.8-VL main model must pair with the Qwen3.8-VL mmproj.\n"
                    "3. Insufficient VRAM caused CLIP encoding to fail - try reducing the number of input images/video frames or restart ComfyUI.\n"
                    "4. The image file is corrupted or in an unusual format - check that the input image opens normally.\n"
                    f"Original error: {e}"
                ) from e
            raise

        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = str(out)

        # Strata HTTP may return None; fall back to an empty string
        if text is None:
            text = ""

        if auto_unload:
            _BSAI_QwenStorage.unload()

        return (text.lstrip().removeprefix(": ").strip(),)


class BSAI_QwenUnloadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any_input": ("*",)}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "run"
    CATEGORY = "BSAI"

    def run(self, any_input):
        _BSAI_QwenStorage.unload()
        return (any_input,)


# ---- BSAI VLP: exact video info endpoint (replaces unreliable RVFC fps guess) ----
try:
    from aiohttp import web as _aiohttp_web
    from server import PromptServer as _PromptServer

    @_PromptServer.instance.routes.get("/bsai_vlp/video_info")
    async def _bsai_vlp_video_info(request):
        fname = os.path.basename(request.query.get("filename", ""))
        subfolder = str(request.query.get("subfolder", "") or "").replace("\\", "/").strip("/")
        if not fname:
            return _aiohttp_web.json_response({"error": "no filename"}, status=400)
        try:
            base = os.path.realpath(folder_paths.get_input_directory())
            vpath = os.path.realpath(os.path.join(base, subfolder, fname))
            if os.path.commonpath([base, vpath]) != base:
                return _aiohttp_web.json_response({"error": "invalid path"}, status=400)
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                return _aiohttp_web.json_response({"error": "cannot open"}, status=404)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if fps <= 0 or fps != fps:  # NaN check
                fps = 30.0
            return _aiohttp_web.json_response({
                "fps": float(fps), "total_frames": total,
                "width": w, "height": h,
                "duration": float(total) / float(fps),
            })
        except Exception as e:
            return _aiohttp_web.json_response({"error": str(e)}, status=500)
except Exception as _e:
    print(f"[BSAI_VLP] video_info route not registered: {_e}")


# ---- BSAI Step Schedule Config: one dropdown drives VedaSparse / Timesteps / EulerSampler ----
_RECIPE_4STEP = "4-step Preview (999,749,500,250 · shift 12/3 · keep 10%)"
_RECIPE_8STEP_V2 = "8-step V2 (999,874,749,624,500,375,250,125 · shift 10/3 · keep 20%)"
_RECIPE_STEP3 = "3-step TaoMate (999,750,500 · shift 12/3 · keep 100% Dense)"
_RECIPE_CUSTOM = "custom"
_RECIPE_LIST = (_RECIPE_4STEP, _RECIPE_8STEP_V2, _RECIPE_STEP3, _RECIPE_CUSTOM)

_RECIPE_LADDER = {
    _RECIPE_4STEP: "999,749,500,250",
    _RECIPE_8STEP_V2: "999,874,749,624,500,375,250,125",
    _RECIPE_STEP3: "999,750,500",
}
_RECIPE_SHIFT_V = {_RECIPE_4STEP: 12.0, _RECIPE_8STEP_V2: 10.0, _RECIPE_STEP3: 12.0}
_RECIPE_SHIFT_A = {_RECIPE_4STEP: 3.0, _RECIPE_8STEP_V2: 3.0, _RECIPE_STEP3: 3.0}
_RECIPE_KEEP = {_RECIPE_4STEP: 10.0, _RECIPE_8STEP_V2: 20.0, _RECIPE_STEP3: 100.0}


class BSAI_StepScheduleConfig(ComfyNodeABC):
    """Single step-schedule dropdown that feeds VedaSparse / Timesteps / EulerSampler.

    Right-click the 'recipe' widget on each FastH3 consumer node → Convert to Input,
    then wire from this node's 'recipe' output.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "recipe": (list(_RECIPE_LIST), {
                    "default": _RECIPE_8STEP_V2,
                    "tooltip": (
                        "One-click step-schedule preset. Changing it drives VedaSparse + Timesteps + EulerSampler together.\n\n"
                        "Must match the base model/LoRA:\n"
                        "• 8-step V2 -> base model fastvideo_fasth3_8step_v2_pruned_int8 (current default)\n"
                        "• 4-step Preview -> enable 4-step LoRA (DMAD/FlashGen/PDMD), strength=1.0\n"
                        "• 3-step TaoMate -> requires the 3-step TaoMate LoRA (not pre-wired)\n"
                        "Mismatched step counts produce blurry/noisy output."
                    ),
                }),
                "custom_ladder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Used when recipe=custom, e.g. 999,749,500,250",
                }),
            },
        }

    RETURN_TYPES = (
        list(_RECIPE_LIST),
        "STRING", "FLOAT", "FLOAT", "FLOAT"
    )
    RETURN_NAMES = ("recipe", "ladder", "shift_video", "shift_audio", "keep_percent")
    FUNCTION = "run"
    CATEGORY = "BSAI"
    OUTPUT_NODE = False
    DESCRIPTION = "Single step-schedule config node: one dropdown drives the recipe parameter of VedaSparse / Timesteps / EulerSampler."

    def run(self, recipe, custom_ladder=""):
        ladder = _RECIPE_LADDER.get(recipe, custom_ladder or "999,749,500,250")
        sv = _RECIPE_SHIFT_V.get(recipe, 12.0)
        sa = _RECIPE_SHIFT_A.get(recipe, 3.0)
        keep = _RECIPE_KEEP.get(recipe, 20.0)
        print(f"[BSAI_StepScheduleConfig] recipe={recipe}", flush=True)
        print(f"  ladder={ladder}  shift_v={sv}  shift_a={sa}  keep={keep}%", flush=True)
        return (recipe, ladder, sv, sa, keep)


NODE_CLASS_MAPPINGS = {
    "BSAI_MultiplePathsInputPlus": BSAI_MultiplePathsInputPlus,
    "BSAI_VideoLoaderPlus": BSAI_VideoLoaderPlus,
    "MultiplePathsInputPlus": BSAI_MultiplePathsInputPlus,
    "VideoLoaderPlus": BSAI_VideoLoaderPlus,
    "BSAI_QwenModelLoader": BSAI_QwenModelLoader,
    "BSAI_QwenPromptInference": BSAI_QwenPromptInference,
    "BSAI_QwenMultimodalInference": BSAI_QwenMultimodalInference,
    "BSAI_QwenUnloadModel": BSAI_QwenUnloadModel,
    "BSAI_StepScheduleConfig": BSAI_StepScheduleConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MultiplePathsInputPlus": "BSAI Multiple Paths Input Plus",
    "BSAI_VideoLoaderPlus": "BSAI Video Loader Plus",
    "BSAI_QwenModelLoader": "BSAI Qwen Model Loader",
    "BSAI_QwenPromptInference": "BSAI Qwen Prompt Inference",
    "BSAI_QwenMultimodalInference": "BSAI Qwen Multimodal Inference",
    "BSAI_QwenUnloadModel": "BSAI Qwen Unload Model",
    "BSAI_StepScheduleConfig": "BSAI Step Schedule Config",
}
