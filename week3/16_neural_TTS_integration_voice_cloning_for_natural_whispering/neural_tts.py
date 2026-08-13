import os
import time
import tempfile
import threading
import numpy as np
import sounddevice as sd
import torch

# NeuralWhisperTTS class with pluggable engines (xtts, pyttsx3).
# Voice cloning through speaker_wav.
# synthesize() returning a float NumPy array.
# speak() and speak_async() for playback.
# A warmup() method to avoid first-call latency.

class NeuralWhisperTTS:
    """Neural text-to-speech with optional voice cloning.

    Engines:
      - xtts: Coqui XTTS-v2 (requires `pip install TTS`, GPU recommended)
      - pyttsx3: offline fallback (robotic but works everywhere)
    """

    def __init__(self, engine="xtts", speaker_wav=None, language="en", device=None):
        self.engine = engine.lower()
        self.speaker_wav = speaker_wav
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 24000
        self.tts = None
        self._pyttsx3_engine = None

        if self.engine == "xtts":
            self._init_xtts()
        elif self.engine == "pyttsx3":
            self._init_pyttsx3()
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")

    def _init_xtts(self):
        try:
            from TTS.api import TTS
            print(f"Loading XTTS-v2 on {self.device}...")
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            ).to(self.device)
            self.sample_rate = getattr(self.tts.synthesizer, "output_sample_rate", 24000)
            print(f"XTTS-v2 loaded. Sample rate: {self.sample_rate}")
        except Exception as e:
            print(f"XTTS init failed: {e}")
            print("Falling back to pyttsx3")
            self.engine = "pyttsx3"
            self._init_pyttsx3()

    def _init_pyttsx3(self):
        import pyttsx3
        self._pyttsx3_engine = pyttsx3.init()
        self.sample_rate = 22050
        print("pyttsx3 fallback ready")

    def set_voice(self, speaker_wav):
        """Update the voice cloning reference for XTTS."""
        self.speaker_wav = speaker_wav
        if self.engine == "xtts":
            print(f"Speaker reference updated: {speaker_wav}")

    def warmup(self):
        """Run a dummy synthesis to load kernels and cache speaker embeddings."""
        if self.engine == "xtts":
            print("Warming up XTTS...")
            _ = self.synthesize("Warm up.")
            print("Warmup complete")

    def synthesize(self, text):
        """Return (wav_float32_numpy, sample_rate)."""
        if self.engine == "xtts":
            wav = self.tts.tts(
                text=text,
                speaker_wav=self.speaker_wav,
                language=self.language,
            )
            if isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)
            elif isinstance(wav, np.ndarray):
                wav = wav.astype(np.float32)
            else:
                raise TypeError(f"Unexpected TTS output type: {type(wav)}")
            return wav, self.sample_rate

        elif self.engine == "pyttsx3":
            return self._synthesize_pyttsx3(text)

        else:
            raise RuntimeError("No TTS engine initialized")

    def _synthesize_pyttsx3(self, text):
        import soundfile as sf
        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        self._pyttsx3_engine.save_to_file(text, tmp_path)
        self._pyttsx3_engine.runAndWait()
        wav, sr = sf.read(tmp_path)
        os.unlink(tmp_path)
        return wav.astype(np.float32), sr

    def speak(self, text, wait=True):
        """Play TTS audio through speakers/earpiece."""
        wav, sr = self.synthesize(text)
        sd.play(wav, sr)
        if wait:
            sd.wait()

    def speak_async(self, text):
        """Non-blocking playback in a daemon thread."""
        thread = threading.Thread(target=self.speak, args=(text, True), daemon=True)
        thread.start()
        return thread