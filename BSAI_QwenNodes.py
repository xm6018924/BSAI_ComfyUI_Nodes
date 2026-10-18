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
        if mmproj and mmproj != "None":
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


class BSAI_QwenModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        all_files = _bsai_list_llm_files()
        model_list = [f for f in all_files if "mmproj" not in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin", ".pth", ".pt"]]
        mmproj_list = ["None"] + [f for f in all_files if "mmproj" in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin"]]

        if not model_list:
            model_list = ["(place models in models/LLM)"]

        # Strata quant options
        strata_quants = ["Auto (auto)", "Q2_0 (fastest)", "IQ2_XS (slightly better)", "IQ3_XXS (better)", "IQ3_S (best)"]

        return {
            "required": {
                "backend": (["llama.cpp (local GGUF)", "BSAI Strata HTTP (Qwen3.8-Flash-Next @127.0.0.1:8080)"],
                          {"default": "llama.cpp (local GGUF)",
                           "tooltip": "Strata backend does not need model/mmproj; it calls the local Strata service over HTTP"}),
                "model_family": (["Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL", "Gemma4"], {"default": "Qwen3.8-VL"}),
                "model_file": (model_list, {"tooltip": "Main model file (recommended .gguf) placed in ComfyUI/models/LLM/"}),
                "mmproj": (mmproj_list, {"default": "None", "tooltip": "Multimodal requires mmproj; for text-only you may select None."}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "context_length": ("INT", {"default": 32768, "min": 1024, "max": 327680, "step": 256}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
                "strata_quant": (strata_quants, {"default": "Auto (auto)", "tooltip": "Strata backend only: select quantization level; Auto picks the fastest"}),
            }
        }

    RETURN_TYPES = ("BSAI_QWEN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "BSAI"

    def load(self, backend, model_family, model_file, mmproj, enable_thinking, context_length, gpu_layers, strata_quant="Auto (auto)"):
        # --- Strata HTTP backend: skip llama.cpp loading entirely ---
        if str(backend).startswith("BSAI Strata"):
            # Parse quant label
            quant_map = {
                "Auto (auto)": None,
                "Q2_0 (fastest)": "Q2_0",
                "IQ2_XS (slightly better)": "IQ2_XS",
                "IQ3_XXS (better)": "IQ3_XXS",
                "IQ3_S (best)": "IQ3_S",
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

            return (_BSAI_StrataHTTPBackend(n_ctx=int(context_length), quant_label=quant),)

        if model_file is None or str(model_file).startswith("(place models"):
            raise RuntimeError("No model file found. Please place the model in ComfyUI/models/LLM/ and restart.")

        if model_family in ("Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL", "Qwen3.8-VL", "Gemma4"):
            if str(mmproj) in ("None",):
                raise RuntimeError(
                    f"{model_family} is a multimodal model and requires an mmproj file.\n" +
                    "Please select the matching mmproj file in the 'mmproj' option."
                )

        config = {
            "family": model_family,
            "model": model_file,
            "mmproj": mmproj,
            "think": bool(enable_thinking),
            "n_ctx": int(context_length),
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
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "BSAI"

    def run(self, qwen_model, text, system_prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, frequency_penalty, presence_penalty, seed, auto_unload, video_max_frames, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, video_1=None, video_2=None, video_3=None):
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
            video_path = _bsai_get_video_path(video_input)
            if video_path:
                frames = _bsai_extract_video_frames(video_path, max_frames=int(video_max_frames))
                for fidx, frame in enumerate(frames):
                    temp_path = temp_dir / f"vid_{int(time.time()*1000)}_{idx}_f{fidx}.jpg"
                    img = Image.fromarray(frame)
                    img.save(temp_path, quality=95)
                    content_items.append({"type": "image_url", "image_url": {"url": Path(str(temp_path)).as_uri()}})

        # Text prompt
        prompt_text = (text or "").strip()
        if not prompt_text and not content_items:
            raise ValueError("Input text and all image/video inputs cannot be empty at the same time.")
        if prompt_text:
            content_items.append({"type": "text", "text": prompt_text})

        # If we are on the Strata HTTP backend, convert local file:// image URLs to
        # base64 data URLs (OpenAI-compatible / Strata mmproj cannot read local paths).
        if isinstance(llm, _BSAI_StrataHTTPBackend):
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
        fname = request.query.get("filename", "")
        subfolder = request.query.get("subfolder", "")
        if not fname:
            return _aiohttp_web.json_response({"error": "no filename"}, status=400)
        try:
            base = folder_paths.get_input_directory()
            vpath = os.path.join(base, subfolder, fname) if subfolder else os.path.join(base, fname)
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
