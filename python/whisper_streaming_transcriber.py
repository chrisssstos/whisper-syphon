"""
Streaming transcription using mlx-whisper.
Handles audio buffering, VAD, and progressive text emission.
"""

import threading
import time
from typing import Callable, Optional

import numpy as np


class WhisperStreamingTranscriber:
    """
    Streaming transcription using mlx-whisper.
    Buffers audio, detects voice activity, and emits text progressively.
    """

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        sample_rate: int = 16000,
        min_chunk_seconds: float = 0.5,
        max_buffer_seconds: float = 30.0,
        vad_threshold: float = 0.35,
        silence_threshold: float = 0.01,
        on_text: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize streaming transcriber.

        Args:
            model_name: Whisper model to use (large-v3-turbo recommended for lyrics)
            sample_rate: Audio sample rate (should be 16kHz for Whisper)
            min_chunk_seconds: Minimum audio to buffer before processing
            max_buffer_seconds: Maximum audio buffer size
            vad_threshold: Voice activity detection threshold
            silence_threshold: RMS threshold for silence detection
            on_text: Callback for emitting transcribed text
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.min_chunk_samples = int(min_chunk_seconds * sample_rate)
        self.max_buffer_samples = int(max_buffer_seconds * sample_rate)
        self.vad_threshold = vad_threshold
        self.silence_threshold = silence_threshold
        self.on_text = on_text

        # Audio buffer
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # State tracking
        self._last_text = ""
        self._last_activity_time = 0
        self._silence_duration = 0
        self._running = False
        self._transcribing = False  # Prevent concurrent transcription
        self._last_transcribe_time = 0
        self._min_transcribe_interval = 1.0  # Minimum seconds between transcriptions

        # Models (initialized in load_model)
        self.model = None
        self.vad = None
        self._vad_ready = False
        self._model_path = f"mlx-community/whisper-{model_name}"

    def load_model(self):
        """Load Whisper model and VAD."""
        print(f"[Whisper] Loading model: {self.model_name}...")

        # Load mlx-whisper module
        import mlx_whisper
        self.model = mlx_whisper

        # Pre-download and cache the model by running a dummy transcription
        # This ensures the model is ready before transcription starts
        self._model_path = f"mlx-community/whisper-{self.model_name}"
        print(f"[Whisper] Downloading/caching model from {self._model_path}...")
        dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
        try:
            self.model.transcribe(
                dummy_audio,
                path_or_hf_repo=self._model_path,
                condition_on_previous_text=False,
            )
            print(f"[Whisper] Model cached and ready: {self.model_name}")
        except Exception as e:
            print(f"[Whisper] Model warmup warning (expected): {e}")

        print(f"[Whisper] Model ready: {self.model_name}")

        # Load Silero VAD for voice activity detection
        try:
            import torch
            self.vad, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._vad_ready = True
            print("[Whisper] VAD loaded successfully")
        except Exception as e:
            print(f"[Whisper] VAD not available: {e}")
            print("[Whisper] Falling back to RMS-based silence detection")
            self._vad_ready = False

    def insert_audio(self, audio: np.ndarray):
        """
        Add audio to buffer.

        Args:
            audio: Float32 audio at 16kHz
        """
        # Flatten if needed
        if audio.ndim > 1:
            audio = audio.flatten()

        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, audio])

            # Trim if buffer too large
            if len(self._buffer) > self.max_buffer_samples:
                self._buffer = self._buffer[-self.max_buffer_samples:]

    def _is_silence(self, audio: np.ndarray) -> bool:
        """Check if audio is silence using RMS."""
        if len(audio) < 100:
            return True
        rms = np.sqrt(np.mean(audio**2))
        return rms < self.silence_threshold

    def _detect_voice_activity(self, audio: np.ndarray) -> bool:
        """Check if audio contains voice using Silero VAD."""
        if not self._vad_ready:
            # Fall back to RMS-based detection
            return not self._is_silence(audio)

        if len(audio) < 512:
            return False

        try:
            import torch
            # VAD expects audio in specific format
            audio_tensor = torch.from_numpy(audio).float()
            speech_prob = self.vad(audio_tensor, self.sample_rate).item()
            return speech_prob > self.vad_threshold
        except Exception:
            return not self._is_silence(audio)

    def process_step(self) -> Optional[str]:
        """
        Process buffered audio and return new text.

        Returns:
            New transcribed text, or None if no new text
        """
        # Prevent concurrent transcription (causes Metal GPU crashes)
        if self._transcribing:
            return None

        # Throttle transcription calls
        current_time = time.time()
        if current_time - self._last_transcribe_time < self._min_transcribe_interval:
            return None

        with self._buffer_lock:
            if len(self._buffer) < self.min_chunk_samples:
                return None
            audio = self._buffer.copy()
            # Clear buffer after copying to prevent re-processing
            self._buffer = np.array([], dtype=np.float32)

        # Check for voice activity in recent audio
        recent_audio = audio[-self.min_chunk_samples:]
        has_voice = self._detect_voice_activity(recent_audio)

        if not has_voice:
            # Track silence duration
            if self._last_activity_time > 0:
                self._silence_duration = current_time - self._last_activity_time

                # Reset context after extended silence to prevent hallucinations
                if self._silence_duration > 3.0:
                    self._last_text = ""
            return None

        self._last_activity_time = current_time
        self._silence_duration = 0

        # Transcribe using mlx-whisper
        self._transcribing = True
        self._last_transcribe_time = current_time
        try:
            result = self.model.transcribe(
                audio,
                path_or_hf_repo=self._model_path,
                condition_on_previous_text=False,  # Prevent hallucinations
                word_timestamps=True,
            )

            text = result.get("text", "").strip()

            if text and text != self._last_text:
                # Find the new portion of text
                if self._last_text and text.startswith(self._last_text):
                    delta = text[len(self._last_text):].strip()
                else:
                    delta = text

                self._last_text = text

                if delta:
                    return delta

        except Exception as e:
            print(f"[Whisper] Transcription error: {e}")
        finally:
            self._transcribing = False

        return None

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._buffer_lock:
            return len(self._buffer) / self.sample_rate

    def reset(self):
        """Reset transcriber state."""
        with self._buffer_lock:
            self._buffer = np.array([], dtype=np.float32)
        self._last_text = ""
        self._last_activity_time = 0
        self._silence_duration = 0
        print("[Whisper] Transcriber reset")
