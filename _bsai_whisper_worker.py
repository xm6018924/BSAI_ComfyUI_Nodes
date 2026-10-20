# -*- coding: utf-8 -*-
"""
BSAI Whisper ASR worker — runs in the XPU/CPU python_embeded interpreter.
Loads the local whisper-large-v3 once, warms up XPU JIT, then stays resident,
reading JSON requests {"wav": "<path>"} from stdin and writing
{"ok": true, "text": "..."} lines to stdout.

Protocol:
  in:   {"wav": "C:/abs/path/to/audio.wav", "language": "zh" | null}
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

# 低于该 RMS(约-49dBFS)的 30s 块视为静音, 直接标注跳过, 不进 Whisper
# (防止长前奏/间奏静音段触发 Whisper 幻觉文本, 同时省推理时间)
_SILENCE_RMS = 0.0035


def _fmt_ts(sec):
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(round(sec - m * 60))
    if s >= 60:
        m += 1
        s -= 60
    return f"{m:02d}:{s:02d}"


def _transcribe_array(audio_16k, language=None):
    """30s 分块转写 -> '[mm:ss-mm:ss] 文本' 分段文本(带时间轴锚点, 供下游 LLM 使用)。

    静音块不推理直接标「（无人声）」; 强制 language 消除语言误判型幻觉;
    转写为空的块标「（无人声/纯音乐）」。language=None 时由 Whisper 自动检测。
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
        ts = f"[{_fmt_ts(t0)}-{_fmt_ts(t1)}]"
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < _SILENCE_RMS:
            segs.append(f"{ts}（无人声）")
            continue
        inputs = proc(chunk, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features.to(DEV).to(model.dtype)
        with torch.no_grad():
            pred = model.generate(feat, **gen_kw)
        text = proc.batch_decode(pred, skip_special_tokens=True)[0].strip()
        segs.append(f"{ts} {text}" if text else f"{ts}（无人声/纯音乐）")
    return "\n".join(segs)


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
        text = _transcribe_array(np.asarray(audio, dtype=np.float32),
                                 language=req.get("language") or None)
        sys.stdout.write(json.dumps({"ok": True, "text": text}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": repr(e)[:500]}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

log("worker exit")
