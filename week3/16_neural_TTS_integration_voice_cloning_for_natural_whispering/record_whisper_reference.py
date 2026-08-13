import sounddevice as sd
import numpy as np
import wave
import sys

# We need a clean 6–10 second whisper sample for XTTS voice cloning.

def record_reference(filename="whisper_ref.wav", duration=10, sr=24000):
    """Record a short whisper sample for XTTS voice cloning."""
    print(f"Recording {duration} seconds of whisper...")
    print("Speak in a whisper, e.g. 'I am your assistant, I will correct you silently.'")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = np.clip(audio, -1.0, 1.0)
    int_audio = (audio * 32767).astype(np.int16)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(int_audio.tobytes())

    print(f"Saved reference to {filename}")

if __name__ == "__main__":
    record_reference()