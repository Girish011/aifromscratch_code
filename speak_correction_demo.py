"""
Speak correction whispers through the system audio device
=========================================================
Bluetooth headsets do NOT plug into MiniLM/spaCy directly.

Pipeline:
  1) Pair headset in macOS System Settings → Bluetooth
  2) Set it as OUTPUT (and INPUT if you want mic) in Sound settings
  3) This script runs TTS → sound goes to whatever device is default
     (your headset, once selected)

macOS: uses the built-in `say` command (no extra pip package).
Optional later: pyttsx3, ElevenLabs, OpenAI TTS, etc.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from correction_engine import LiveCorrectionEngine


def speak(text: str, *, rate: int = 160, voice: str | None = None) -> None:
    """
    Speak `text` via macOS `say`.
    rate: words-ish per minute (lower ≈ softer / more 'whisper-like' pacing)
    voice: e.g. 'Samantha', 'Alex' — run `say -v '?'` to list
    """
    if not text or not text.strip():
        return
    if shutil.which("say") is None:
        raise RuntimeError(
            "macOS `say` not found. On Linux/Windows use pyttsx3 or another TTS."
        )
    cmd = ["say", "-r", str(rate)]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    subprocess.run(cmd, check=False)


def demo_once(ground_truth: str, user_utterance: str) -> dict:
    engine = LiveCorrectionEngine()
    result = engine.process(ground_truth, user_utterance)
    print("similarity:", round(result["similarity"], 3))
    print("interrupt:", result["needs_interrupt"])
    print("whisper:", result["whisper"])

    if result["needs_interrupt"]:
        # Only speak when a correction is warranted
        speak(result["whisper"], rate=150)
    else:
        print("(no spoken whisper — aligned with source)")
    return result


if __name__ == "__main__":
    # Default demo; pass custom pairs via editing or extend CLI as you like
    truth = "Our AI platform reduces costs by 40%"
    user = "Our AI platform increases costs by 30%"
    if len(sys.argv) >= 3:
        truth, user = sys.argv[1], sys.argv[2]
    demo_once(truth, user)
