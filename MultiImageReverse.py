from __future__ import annotations
import copy
import random
import re
import time

from ollama import Client
import numpy as np
import base64
from io import BytesIO
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass, field
from pydantic.json_schema import JsonSchemaValue


def _mir_ninfer_model_list():
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


def _mir_nemotron_model_list():
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

# For type checking only. Torch is not installed at runtime
if TYPE_CHECKING:
    import torch

# 尝试导入ComfyUI的server模块，如果失败则跳过（用于独立测试）
try:
    from server import PromptServer
    from aiohttp import web
    import urllib.parse as _urlparse
    import socket as _socket

    # Security: only allow loopback and private-LAN hosts for Ollama/Strata
    # requests, preventing SSRF through a user-supplied url widget.
    def _is_allowed_host(url):
        try:
            parts = _urlparse.urlparse(str(url or "").strip())
            host = (parts.hostname or "").strip().lower()
            if not host:
                return False
            if host in ("localhost", "127.0.0.1", "::1", "[::1]"):
                return True
            try:
                ip = _socket.gethostbyname(host)
            except Exception:
                return False
            if ip.startswith("127.") or ip == "::1":
                return True
            # RFC1918 private ranges + loopback
            if ip.startswith("10.") or ip.startswith("192.168."):
                return True
            if ip.startswith("172."):
                try:
                    return 16 <= int(ip.split(".")[1]) <= 31
                except Exception:
                    return False
            return False
        except Exception:
            return False

    @PromptServer.instance.routes.post("/bsai_ollama/get_models")
    async def get_models_endpoint(request):
        data = await request.json()

        url = data.get("url")
        if not _is_allowed_host(url):
            return web.json_response(
                {"error": "Ollama host must be localhost or a private LAN address (SSRF guard)"},
                status=400,
            )
        client = Client(host=url)

        models = client.list().get('models', [])

        try:
            models = [model['model'] for model in models]
            return web.json_response(models)
        except Exception as e:
            models = [model['name'] for model in models]
            return web.json_response(models)
except Exception:
    # 独立测试或 PromptServer 未就绪时跳过 server 相关代码
    pass

class BSAI_MultiImageVideoReverse:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["Ollama (local 11434)", "BSAI Strata (Qwen3.8-Flash-Next @8080)", "BSAI Nemotron (llama.cpp @8085)", "BSAI NInfer (本地反推 @808x)"],
                            {"default": "Ollama (local 11434)",
                             "tooltip": "Inference backend: Ollama, BSAI Strata HTTP, or BSAI NInfer (AI蛮子 ninfer engine, dynamic free port)"}),
                "strata_quant": (["Auto (auto)", "Q2_0 (fastest)", "IQ2_XS (slightly better)", "IQ3_XXS (better)", "IQ3_S (best)"],
                            {"default": "Auto (auto)", "tooltip": "Strata backend only: select quantization level, Auto picks the fastest"}),
                "ninfer_model": (["(Auto — first model found)"] + _mir_ninfer_model_list(),
                            {"default": "(Auto — first model found)",
                             "tooltip": "NInfer backend only: which *.ninfer model to load (models/LLM). Changing it restarts the NInfer engine."}),
                "nemotron_model": (["(Auto — first model found)"] + _mir_nemotron_model_list(),
                            {"default": "(Auto — first model found)",
                             "tooltip": "Nemotron backend only: which GGUF model to load (models/LLM/Nemotron, excludes mmproj). Changing it restarts llama.cpp."}),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an expert image and video analyst, skilled at analyzing multiple images and video frames to extract common features and themes. Analyze the following content in detail, extract key elements, style, color, composition, and generate a detailed description.",
                    "tooltip": "System prompt - sets the model role and behavior"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Analyze these images and video frames in detail, including all details, and generate a comprehensive description.",
                    "tooltip": "User prompt - the task you want the model to perform"
                }),
            },
            "optional": {
                "model": (["qwen3:latest", "qwen3.5:latest", "qwen3.6:latest", "qwen3.8:latest", "gemma4:31b", "custom"],
                          {"default": "qwen3.8:latest", "tooltip": "Ollama model selection (Ollama backend only)"}),
                "url": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "Ollama server URL (Ollama backend only)"
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Custom model name (used when custom is selected)"
                }),
                # 图像输入（最多10张）
                "image1": ("IMAGE", {"forceInput": False, "tooltip": "First input image"}),
                "image2": ("IMAGE", {"forceInput": False, "tooltip": "Second input image"}),
                "image3": ("IMAGE", {"forceInput": False, "tooltip": "Third input image"}),
                "image4": ("IMAGE", {"forceInput": False, "tooltip": "Fourth input image"}),
                "image5": ("IMAGE", {"forceInput": False, "tooltip": "Fifth input image"}),
                "image6": ("IMAGE", {"forceInput": False, "tooltip": "Sixth input image"}),
                "image7": ("IMAGE", {"forceInput": False, "tooltip": "Seventh input image"}),
                "image8": ("IMAGE", {"forceInput": False, "tooltip": "Eighth input image"}),
                "image9": ("IMAGE", {"forceInput": False, "tooltip": "Ninth input image"}),
                "image10": ("IMAGE", {"forceInput": False, "tooltip": "Tenth input image"}),
                # 视频输入（最多5个视频）
                "video1": ("IMAGE", {"forceInput": False, "tooltip": "First video (frame sequence)"}),
                "video2": ("IMAGE", {"forceInput": False, "tooltip": "Second video (frame sequence)"}),
                "video3": ("IMAGE", {"forceInput": False, "tooltip": "Third video (frame sequence)"}),
                "video4": ("IMAGE", {"forceInput": False, "tooltip": "Fourth video (frame sequence)"}),
                "video5": ("IMAGE", {"forceInput": False, "tooltip": "Fifth video (frame sequence)"}),
                # 视频采样参数
                "video_sample_interval": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, 
                                                  "tooltip": "Video frame sampling interval (take one frame every N frames)"}),
                "max_frames_per_video": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, 
                                                 "tooltip": "Maximum number of frames sampled per video"}),
                # 推理参数
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05, "tooltip": "Controls text creativity; higher = more creative"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "Controls text diversity; higher = more varied"}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 100000, "step": 1, "tooltip": "Maximum number of tokens to generate"}),
            }
        }

    # 允许用户手动添加更多输入端口
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return str(time.time())

    RETURN_TYPES = ("STRING", "STRING", "*")
    RETURN_NAMES = ("response", "thinking", "optional_prompt_list")
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "multi_media_reverse"
    CATEGORY = "BSAI"
    DESCRIPTION = "BSAI multi-image/video reverse prompt node: uses a local Ollama model to analyze multiple images and video frames and generate a description"

    def _process_image_tensor(self, img_tensor):
        """Process a single image tensor, convert to base64."""
        images_b64 = []
        try:
            # Check whether this is a torch tensor
            if hasattr(img_tensor, 'cpu'):
                print(f"[BSAI_MultiImageVideoReverse] Processing image, shape: {img_tensor.shape}")
                # Batch images (4D tensor: B, H, W, C)
                if len(img_tensor.shape) == 4:
                    for idx in range(img_tensor.shape[0]):
                        img_np = 255. * img_tensor[idx].cpu().numpy()
                        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        images_b64.append(img_bytes)
                        print(f"[BSAI_MultiImageVideoReverse] Processed batch image {idx+1}/{img_tensor.shape[0]}")
                elif len(img_tensor.shape) == 3:
                    # Single image (3D tensor: H, W, C)
                    img_np = 255. * img_tensor.cpu().numpy()
                    img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    images_b64.append(img_bytes)
                    print(f"[BSAI_MultiImageVideoReverse] Processed single image")
                else:
                    print(f"[BSAI_MultiImageVideoReverse] WARNING: unsupported image tensor dimensions: {img_tensor.shape}")
            else:
                # Not a torch tensor: try to process directly
                print(f"[BSAI_MultiImageVideoReverse] WARNING: input is not a torch tensor, skipping")
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()

        return images_b64

    def _process_video_tensor(self, video_tensor, sample_interval=10, max_frames=5):
        """Process a video tensor, sample key frames and convert to base64."""
        images_b64 = []
        try:
            if video_tensor is None:
                return images_b64

            if hasattr(video_tensor, 'cpu'):
                print(f"[BSAI_MultiImageVideoReverse] Processing video, shape: {video_tensor.shape}")
                # Video is a 4D tensor: B, H, W, C (B = number of frames)
                if len(video_tensor.shape) == 4:
                    total_frames = video_tensor.shape[0]
                    # Compute sampling indices
                    indices = list(range(0, total_frames, sample_interval))[:max_frames]
                    print(f"[BSAI_MultiImageVideoReverse] Video has {total_frames} frames, sampling {len(indices)}: {indices}")

                    for idx in indices:
                        img_np = 255. * video_tensor[idx].cpu().numpy()
                        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        images_b64.append(img_bytes)
                    print(f"[BSAI_MultiImageVideoReverse] Processed sampled video frames")
                else:
                    print(f"[BSAI_MultiImageVideoReverse] WARNING: unsupported video tensor dimensions: {video_tensor.shape}")
            else:
                print(f"[BSAI_MultiImageVideoReverse] WARNING: input is not a torch tensor, skipping")
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()

        return images_b64

    def multi_media_reverse(self, backend, strata_quant, ninfer_model, nemotron_model, system, prompt, model="qwen3.8:latest", url="http://127.0.0.1:11434", custom_model="",
                          image1=None, image2=None, image3=None, image4=None, image5=None,
                          image6=None, image7=None, image8=None, image9=None, image10=None,
                          video1=None, video2=None, video3=None, video4=None, video5=None,
                          video_sample_interval=10, max_frames_per_video=5,
                          temperature=0.7, top_p=0.9, max_tokens=32768):

        # ---- BSAI Strata HTTP backend (OpenAI-compatible) ----
        if str(backend).lower().startswith("bsai strata"):
            return self._reverse_via_strata(
                system, prompt, image1, image2, image3, image4, image5,
                image6, image7, image8, image9, image10,
                video1, video2, video3, video4, video5,
                video_sample_interval, max_frames_per_video,
                temperature, top_p, max_tokens,
                quant_label=strata_quant,
            )

        # ---- BSAI Nemotron HTTP backend (llama.cpp @8085) ----
        if str(backend).lower().startswith("bsai nemotron"):
            return self._reverse_via_nemotron(
                system=system, prompt=prompt,
                image1=image1, image2=image2, image3=image3, image4=image4, image5=image5,
                image6=image6, image7=image7, image8=image8, image9=image9, image10=image10,
                video1=video1, video2=video2, video3=video3, video4=video4, video5=video5,
                video_sample_interval=video_sample_interval,
                max_frames_per_video=max_frames_per_video,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                model_path=nemotron_model,
            )

        # ---- BSAI NInfer HTTP backend (AI蛮子 ninfer engine) ----
        if str(backend).lower().startswith("bsai ninfer"):
            return self._reverse_via_ninfer(
                system, prompt, image1, image2, image3, image4, image5,
                image6, image7, image8, image9, image10,
                video1, video2, video3, video4, video5,
                video_sample_interval, max_frames_per_video,
                temperature, top_p, max_tokens,
                model_path=ninfer_model,
            )

        # Determine the model to use
        if model == "custom" and custom_model:
            selected_model = custom_model
        else:
            selected_model = model

        try:
            if not _is_allowed_host(url):
                error_msg = "Ollama host must be localhost or a private LAN address (SSRF guard)"
                print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
                return error_msg, "", []
            client = Client(host=url)
        except Exception as e:
            error_msg = f"Failed to create Ollama client: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            return error_msg, "", []

        # Collect all images
        all_images_b64 = []

        # Process image inputs
        images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        for i, img in enumerate(images, 1):
            if img is not None:
                print(f"[BSAI_MultiImageVideoReverse] Processing image {i}")
                all_images_b64.extend(self._process_image_tensor(img))

        # Process video inputs
        videos = [video1, video2, video3, video4, video5]
        for i, vid in enumerate(videos, 1):
            if vid is not None:
                print(f"[BSAI_MultiImageVideoReverse] Processing video {i}")
                all_images_b64.extend(self._process_video_tensor(vid, video_sample_interval, max_frames_per_video))

        print(f"[BSAI_MultiImageVideoReverse] Media processing done, {len(all_images_b64)} images/frames total")

        # Check whether we have any images
        if not all_images_b64:
            error_msg = "No valid image or video input; please connect at least one image or one video"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            return error_msg, "", []

        # Build request options
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens
        }

        print(f"[BSAI_MultiImageVideoReverse] Sending request to Ollama, model: {selected_model}")

        # Send request to Ollama
        try:
            response = client.generate(
                model=selected_model,
                system=system,
                prompt=prompt,
                images=all_images_b64,
                options=options,
                keep_alive="5m"
            )

            print(f"[BSAI_MultiImageVideoReverse] Received Ollama response")

            # Extract the response
            ollama_response_text = response.get('response', '')
            ollama_response_thinking = response.get('thinking', '') if 'thinking' in response else ""

            print(f"[BSAI_MultiImageVideoReverse] Response length: {len(ollama_response_text)}")

            prompt_list = [item.strip() for item in ollama_response_text.split('\n') if item.strip()]

            return ollama_response_text, ollama_response_thinking, prompt_list

        except Exception as e:
            error_msg = f"Ollama API call failed: {str(e)}"
            print(f"[BSAI_MultiImageVideoReverse] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg, "", []

    def _reverse_via_strata(self, system, prompt,
                            image1, image2, image3, image4, image5,
                            image6, image7, image8, image9, image10,
                            video1, video2, video3, video4, video5,
                            video_sample_interval, max_frames_per_video,
                            temperature, top_p, max_tokens,
                            quant_label=None):
        """OpenAI-compatible chat.completions call to local Strata (Qwen3.8-Flash-Next)."""
        import json as _json

        # Auto-start Strata if not running
        try:
            import importlib.util as _ilu
            _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
            if os.path.exists(_p):
                _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
                _m = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_m)
                ok, msg = _m.StrataManager.ensure_running(quant_label=quant_label)
                print(f"[BSAI_MultiImageReverse] Strata auto-start: {ok} ({msg})")
        except Exception as _e:
            print(f"[BSAI_MultiImageReverse] Strata auto-start skipped: {_e}")

        # collect all images (already base64 PNG)
        images_b64 = []
        for img in [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]:
            if img is not None:
                images_b64.extend(self._process_image_tensor(img))
        for vid in [video1, video2, video3, video4, video5]:
            if vid is not None:
                images_b64.extend(self._process_video_tensor(vid, video_sample_interval, max_frames_per_video))

        if not images_b64:
            return "No valid image or video input", "", []

        # build OpenAI multimodal content
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})

        body = {
            "model": "qwen3.8-flash-next",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        try:
            import socket as _sk, http.client as _hc
            s = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
            s.settimeout(600); s.setblocking(True)
            s.connect(("127.0.0.1", 8080))
            try:
                conn = _hc.HTTPConnection("127.0.0.1", 8080, timeout=600)
                conn.sock = s
                conn.request("POST", "/v1/chat/completions",
                             body=_json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer strata"})
                out = _json.loads(conn.getresponse().read().decode("utf-8", "replace"))
            finally:
                s.close()
            text = out["choices"][0]["message"]["content"]
            thinking = ""
            prompt_list = [l.strip() for l in text.split("\n") if l.strip()]
            return text, thinking, prompt_list
        except Exception as e:
            err = f"[BSAI Strata reverse failed] {type(e).__name__}: {e} (is Strata running at 127.0.0.1:8080?)"
            print(err)
            return err, "", []

    def _reverse_via_nemotron(self, system, prompt,
                              image1, image2, image3, image4, image5,
                              image6, image7, image8, image9, image10,
                              video1, video2, video3, video4, video5,
                              video_sample_interval, max_frames_per_video,
                              temperature, top_p, max_tokens, model_path="(Auto — first model found)"):
        """OpenAI-compatible chat.completions call to local Nemotron 3.5 Bolt
        (llama.cpp @127.0.0.1:8085, auto-started via NemotronManager)."""
        import json as _json

        # Auto-start Nemotron if not running
        try:
            import importlib.util as _ilu
            _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
            if os.path.exists(_p):
                _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
                _m = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_m)
                _mp = None if str(model_path).startswith("(Auto") else str(model_path)
                ok, msg = _m.NemotronManager.ensure_running(model_path=_mp)
                print(f"[BSAI_MultiImageReverse] Nemotron auto-start: {ok} ({msg})")
        except Exception as _e:
            print(f"[BSAI_MultiImageReverse] Nemotron auto-start skipped: {_e}")

        # collect all images (already base64 PNG)
        images_b64 = []
        for img in [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]:
            if img is not None:
                images_b64.extend(self._process_image_tensor(img))
        for vid in [video1, video2, video3, video4, video5]:
            if vid is not None:
                images_b64.extend(self._process_video_tensor(vid, video_sample_interval, max_frames_per_video))

        if not images_b64:
            return "No valid image or video input", "", []

        # build OpenAI multimodal content
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})

        body = {
            "model": "Nemotron-3.5-Bolt-74B-A9B",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        try:
            import socket as _sk, http.client as _hc
            s = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
            s.settimeout(600); s.setblocking(True)
            s.connect(("127.0.0.1", 8085))
            try:
                conn = _hc.HTTPConnection("127.0.0.1", 8085, timeout=600)
                conn.sock = s
                conn.request("POST", "/v1/chat/completions",
                             body=_json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer strata"})
                out = _json.loads(conn.getresponse().read().decode("utf-8", "replace"))
            finally:
                s.close()
            text = out["choices"][0]["message"]["content"]
            thinking = ""
            prompt_list = [l.strip() for l in text.split("\n") if l.strip()]
            return text, thinking, prompt_list
        except Exception as e:
            err = f"[BSAI Nemotron reverse failed] {type(e).__name__}: {e} (is llama.cpp running at 127.0.0.1:8085?)"
            print(err)
            return err, "", []

    def _reverse_via_ninfer(self, system, prompt,
                            image1, image2, image3, image4, image5,
                            image6, image7, image8, image9, image10,
                            video1, video2, video3, video4, video5,
                            video_sample_interval, max_frames_per_video,
                            temperature, top_p, max_tokens,
                            model_path="(Auto — first model found)"):
        """OpenAI-compatible chat.completions call to local NInfer (AI蛮子 ninfer engine,
        dynamic port managed by NInferManager in BSAI-LLM-Strata)."""
        import json as _json

        # Auto-start NInfer if not running
        try:
            import importlib.util as _ilu
            _p = os.path.join(os.path.dirname(__file__), "..", "BSAI-LLM-Strata", "strata_manager.py")
            if os.path.exists(_p):
                _spec = _ilu.spec_from_file_location("bsai_strata_manager", _p)
                _m = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_m)
                _mp = None if str(model_path).startswith("(Auto") else str(model_path)
                ok, msg = _m.NInferManager.ensure_running(model_path=_mp)
                print(f"[BSAI_MultiImageReverse] NInfer auto-start: {ok} ({msg})")
                if not ok:
                    return f"[BSAI NInfer reverse failed] 启动失败: {msg}", "", []
        except Exception as _e:
            print(f"[BSAI_MultiImageReverse] NInfer auto-start skipped: {_e}")

        # collect all images (already base64 PNG)
        images_b64 = []
        for img in [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]:
            if img is not None:
                images_b64.extend(self._process_image_tensor(img))
        for vid in [video1, video2, video3, video4, video5]:
            if vid is not None:
                images_b64.extend(self._process_video_tensor(vid, video_sample_interval, max_frames_per_video))

        if not images_b64:
            return "No valid image or video input", "", []

        # build OpenAI multimodal content
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})

        body = {
            "model": "default",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        try:
            import socket as _sk, http.client as _hc
            from urllib.parse import urlparse as _up
            u = _up(_m.NInferManager.base_url())
            mids = _m.NInferManager.models()
            if mids:
                body["model"] = mids[0]
            s = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
            s.settimeout(600); s.setblocking(True)
            s.connect((u.hostname, u.port or 80))
            try:
                conn = _hc.HTTPConnection(u.hostname, u.port or 80, timeout=600)
                conn.sock = s
                conn.request("POST", u.path + "/chat/completions",
                             body=_json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json"})
                out = _json.loads(conn.getresponse().read().decode("utf-8", "replace"))
            finally:
                s.close()
            text = out["choices"][0]["message"]["content"]
            thinking = ""
            prompt_list = [l.strip() for l in text.split("\n") if l.strip()]
            return text, thinking, prompt_list
        except Exception as e:
            err = f"[BSAI NInfer reverse failed] {type(e).__name__}: {e} (is NInfer running?)"
            print(err)
            return err, "", []


NODE_CLASS_MAPPINGS = {
    "BSAI_MultiImageVideoReverse": BSAI_MultiImageVideoReverse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MultiImageVideoReverse": "BSAI Multi-Image Video Reverse",
}
