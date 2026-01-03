"""Whisper transcription engine."""

from faster_whisper import WhisperModel
import numpy as np
from typing import List, Dict, Optional


class WhisperEngine:
    """Wrapper for faster-whisper model."""

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8"):
        """Initialize Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
            device: Device to run on (cpu, cuda).
            compute_type: Compute type (int8, float16, float32).
        """
        print(f"Loading Whisper model '{model_size}'...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        print("Whisper model loaded")

    def transcribe(self, audio: np.ndarray) -> List[Dict]:
        """Transcribe audio and return words with timestamps.

        Args:
            audio: Audio data as float32 numpy array (16kHz mono).

        Returns:
            List of word dicts with 'text', 'start', 'end' keys.
        """
        if len(audio) == 0:
            return []

        segments, info = self.model.transcribe(
            audio,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )

        words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words.append({
                        'text': word.word.strip(),
                        'start': word.start,
                        'end': word.end
                    })

        return words
