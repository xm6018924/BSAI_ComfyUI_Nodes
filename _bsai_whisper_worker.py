# -*- coding: utf-8 -*-
"""
BSAI Whisper ASR worker — runs in the XPU/CPU python_embeded interpreter.
Loads the local whisper-large-v3 once, warms up XPU JIT, then stays resident,
reading JSON requests {"wav": "<path>"} from stdin and writing
{"ok": true, "text": "..."} lines to stdout.

Protocol:
  in:   {"wav": "C:/abs/path/to/audio.wav"}
        {"wav": "__shutdown__"}
  out:  {"ok": true, "text": "transcript"}
        {"ok": false, "error": "..."}
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")

def log(*a):
    print("[bsai-whisper-worker]", *a, file=sys.stderr, flush=True)

import numpy as np
import soundfile as sf
import torch

# pick device: XPU (Intel iGPU) preferred, CPU fallback
if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
    DEV = "xpu"
else:
    DEV = "cpu"
log("device:", DEV)

from transformers import WhisperProcessor, WhisperForConditionalGeneration

_HERE = os.path.dirname(os.path.abspath(__file__))
MDL_DIR = os.path.realpath(os.path.join(_HERE, "..", "..", "models", "whisper", "whisper-large-v3"))
log("model dir:", MDL_DIR)

proc = WhisperProcessor.from_pretrained(MDL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MDL_DIR).to(DEV).eval()
log("model loaded")


def _transcribe_array(audio_16k):
    win = 30 * 16000
    texts = []
    for i in range(0, len(audio_16k), win):
        chunk = audio_16k[i:i + win]
        if len(chunk) < 8000:  # skip <0.5s tail
            continue
        inputs = proc(chunk, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features.to(DEV).to(model.dtype)
        with torch.no_grad():
            pred = model.generate(feat)
        texts.append(proc.batch_decode(pred, skip_special_tokens=True)[0].strip())
    return " ".join(t for t in texts if t)


# ---- warmup: trigger XPU kernel JIT compilation so the first real request is fast ----
try:
    _warm = np.zeros(16000, dtype=np.float32)
    _inputs = proc(_warm, sampling_rate=16000, return_tensors="pt")
    _feat = _inputs.input_features.to(DEV).to(model.dtype)
    with torch.no_grad():
        _ = model.generate(_feat)
    log("warmup done")
except Exception as e:
    log("warmup failed (non-fatal):", repr(e)[:200])

log("READY")

# ---- main loop ----
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        wav_path = req.get("wav", "")
        if wav_path == "__shutdown__":
            break
        if not wav_path or not os.path.exists(wav_path):
            sys.stdout.write(json.dumps({"ok": False, "error": f"wav not found: {wav_path}"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        text = _transcribe_array(np.asarray(audio, dtype=np.float32))
        sys.stdout.write(json.dumps({"ok": True, "text": text}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": repr(e)[:500]}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

log("worker exit")
