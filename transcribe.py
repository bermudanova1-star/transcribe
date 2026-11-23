#!/usr/bin/env python3
"""Transcribes the audio files listed in jobs/<id>/manifest.json and writes one JSON file.

Same model as the phone app (Moonshine Tiny, English, word timings). Falls back to Whisper tiny.en if Moonshine is unavailable.
Output: {"version": 1, "engine": "...", "clips": [{"index", "name", "durationSec", "transcript": {"text", "lines": [...]}}]}
On any failure it still writes {"error": "..."} so the app can show a clear message.
"""
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

SR = 16000


def log(*a):
    print(*a, flush=True)


def decode(path):
    """Any audio format (mp3, m4a, wav, ogg, flac...) -> mono 16 kHz float32."""
    out = os.path.join(tempfile.mkdtemp(), "a.wav")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", out],
        check=True,
    )
    with wave.open(out, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def find_segments(x, sr=SR):
    """Same pause-bounded chunking as the phone app: pieces of about 26 s cut where the voice is quietest."""
    total = len(x) / sr
    if total <= 34:
        return [(0.0, total)]
    frame = 0.02
    step = max(1, int(round(sr * frame)))
    n = int(np.ceil(len(x) / step))
    rms = np.zeros(n, dtype=np.float32)
    for f in range(n):
        s = x[f * step:(f + 1) * step]
        rms[f] = float(np.sqrt(np.mean(s * s))) if len(s) else 0.0
    ceiling = 0.015
    segs, cursor = [], 0.0
    while cursor < total - 0.1:
        if total - cursor <= 32:
            segs.append((cursor, total))
            break
        target = cursor + 26
        lo = max(cursor + 12, target - 5)
        hi = min(total - 2, target + 5)
        f_lo = max(0, int(lo / frame))
        f_hi = min(n - 1, int(np.ceil(hi / frame)))
        best, best_score = int(round(target / frame)), float("inf")
        for f in range(f_lo, f_hi + 1):
            e = float(rms[f])
            score = e + abs(f * frame - target) * 0.002 + (-0.02 if e <= ceiling else 0.0)
            if score < best_score:
                best, best_score = f, score
        cut = max(cursor + 8, min(total - 1, best * frame))
        segs.append((cursor, cut))
        cursor = cut
    return segs


def pick(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


# ---------------- Moonshine (the model the Android app runs) ----------------
def load_moonshine():
    import moonshine_voice as mv

    arch = None
    for name in ("TINY_STREAMING", "TinyStreaming", "TINY"):
        if hasattr(mv.ModelArch, name):
            arch = getattr(mv.ModelArch, name)
            break
    try:
        model_path, model_arch = mv.get_model_for_language("en", arch) if arch is not None else mv.get_model_for_language("en")
    except TypeError:
        model_path, model_arch = mv.get_model_for_language("en")
    try:
        return mv.Transcriber(model_path=model_path, model_arch=model_arch, options={"word_timestamps": "true"})
    except TypeError:
        return mv.Transcriber(model_path=model_path, model_arch=model_arch)


def moonshine_lines(engine, chunk, offset):
    result = engine.transcribe_without_streaming(chunk.tolist(), SR)
    lines = []
    for ln in pick(result, "lines", default=[]) or []:
        text = " ".join(str(pick(ln, "text", default="")).split())
        if not text:
            continue
        words = []
        for w in pick(ln, "words", default=[]) or []:
            wt = str(pick(w, "word", "text", default="")).strip()
            if not wt:
                continue
            ws = float(pick(w, "start", "start_time", default=0.0) or 0.0) + offset
            we = float(pick(w, "end", "end_time", default=ws - offset) or 0.0) + offset
            words.append({"text": wt, "start": round(ws, 3), "end": round(max(we, ws + 0.02), 3), "confidence": float(pick(w, "confidence", default=0.85) or 0.85)})
        start = float(pick(ln, "start_time", "startTime", default=0.0) or 0.0) + offset
        dur = float(pick(ln, "duration", default=0.4) or 0.4)
        lines.append({"text": text, "startTime": round(max(0.0, start), 3), "duration": round(max(0.08, dur), 3), "words": words})
    return lines


# ---------------- Whisper tiny.en (fallback) ----------------
def load_whisper():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)
    from faster_whisper import WhisperModel

    return WhisperModel("tiny.en", device="cpu", compute_type="int8")


def whisper_lines(model, chunk, offset):
    segs, _ = model.transcribe(chunk, language="en", word_timestamps=True, beam_size=1, vad_filter=False)
    lines = []
    for s in segs:
        text = " ".join(s.text.split())
        if not text:
            continue
        words = [
            {"text": w.word.strip(), "start": round(w.start + offset, 3), "end": round(max(w.end, w.start + 0.02) + offset, 3), "confidence": float(getattr(w, "probability", 0.85) or 0.85)}
            for w in (s.words or [])
            if w.word.strip()
        ]
        lines.append({"text": text, "startTime": round(s.start + offset, 3), "duration": round(max(0.08, s.end - s.start), 3), "words": words})
    return lines


def run_engine(name, loader, fn, clips_audio):
    log("engine:", name)
    engine = loader()
    out = []
    for idx, x in enumerate(clips_audio):
        lines = []
        for (a, b) in find_segments(x):
            chunk = x[int(a * SR):int(b * SR)]
            if len(chunk) >= int(SR * 0.15):
                lines.extend(fn(engine, chunk, a))
            log(f"clip {idx + 1}: transcribed up to {b:.0f}s")
        out.append(lines)
    return out


def warm():
    """Makes sure the speech model files are on disk (they are then saved in the Actions cache) and that the engine starts."""
    ok = False
    try:
        engine = load_moonshine()
        engine.transcribe_without_streaming([0.0] * SR, SR)
        log("moonshine ready")
        ok = True
    except Exception as e:  # noqa: BLE001
        log("Moonshine could not be warmed:", repr(e))
    if not ok:
        try:
            model = load_whisper()
            list(model.transcribe(np.zeros(SR, dtype=np.float32), language="en", beam_size=1)[0])
            log("whisper ready")
        except Exception as e:  # noqa: BLE001
            log("Whisper could not be warmed:", repr(e))
            sys.exit(1)


def main(job_dir, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        with open(os.path.join(job_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        clips = manifest["clips"]
        audio = [decode(os.path.join(job_dir, c["path"])) for c in clips]
        for c, x in zip(clips, audio):
            log(f"{c['name']}: {len(x) / SR:.1f}s of audio")

        engine_name, all_lines = None, None
        try:
            engine_name, all_lines = "moonshine-tiny-streaming-en", run_engine("moonshine", load_moonshine, moonshine_lines, audio)
            if not any(l for l in all_lines):
                raise RuntimeError("Moonshine returned no text")
        except Exception as e:  # noqa: BLE001 - any failure here means: use the fallback
            log("Moonshine could not be used:", repr(e))
            engine_name, all_lines = "whisper-tiny-en", run_engine("whisper", load_whisper, whisper_lines, audio)

        result = {"version": 1, "engine": engine_name, "clips": []}
        for c, x, lines in zip(clips, audio, all_lines):
            result["clips"].append({
                "index": c["index"],
                "name": c["name"],
                "durationSec": round(len(x) / SR, 3),
                "transcript": {"text": " ".join(l["text"] for l in lines).strip(), "lines": lines},
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        log("done")
    except Exception as e:  # noqa: BLE001
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "error": f"{type(e).__name__}: {e}"}, f)
        log("failed:", repr(e))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--warm":
        warm()
    else:
        main(sys.argv[1], sys.argv[2])
