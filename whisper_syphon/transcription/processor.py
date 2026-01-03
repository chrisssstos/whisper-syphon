"""Streaming processor for real-time word extraction."""

import numpy as np
from queue import Queue, Empty
from typing import Optional, List, Dict
import threading
import time

from .engine import WhisperEngine


class StreamingProcessor:
    """Processes audio stream and extracts words in real-time."""

    SAMPLE_RATE = 16000
    BUFFER_DURATION = 2.0  # seconds of audio to accumulate
    OVERLAP_DURATION = 0.5  # seconds of overlap between chunks

    def __init__(self, audio_queue: Queue, word_queue: Queue,
                 model_size: str = "base"):
        """Initialize streaming processor.

        Args:
            audio_queue: Queue receiving audio chunks.
            word_queue: Queue to push extracted words to.
            model_size: Whisper model size.
        """
        self.audio_queue = audio_queue
        self.word_queue = word_queue
        self.engine: Optional[WhisperEngine] = None
        self.model_size = model_size

        self.running = False
        self._thread: Optional[threading.Thread] = None

        # Audio buffer
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_samples = int(self.BUFFER_DURATION * self.SAMPLE_RATE)
        self.overlap_samples = int(self.OVERLAP_DURATION * self.SAMPLE_RATE)

        # Track processed words to avoid duplicates
        self._last_words: List[str] = []

    def start(self) -> None:
        """Start the processing thread."""
        if self.running:
            return

        # Load model
        if self.engine is None:
            self.engine = WhisperEngine(self.model_size)

        self.running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("Streaming processor started")

    def stop(self) -> None:
        """Stop the processing thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("Streaming processor stopped")

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self.running:
            try:
                # Collect audio chunks
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    self.buffer = np.concatenate([self.buffer, chunk])
                except Empty:
                    continue

                # Process when buffer is full enough
                if len(self.buffer) >= self.buffer_samples:
                    self._process_buffer()

            except Exception as e:
                print(f"Processor error: {e}")
                time.sleep(0.1)

    def _process_buffer(self) -> None:
        """Process the audio buffer and extract new words."""
        # Take buffer for processing
        audio = self.buffer[:self.buffer_samples].copy()

        # Keep overlap for continuity
        self.buffer = self.buffer[self.buffer_samples - self.overlap_samples:]

        # Transcribe
        words = self.engine.transcribe(audio)

        # Filter out duplicates and push new words
        for word_info in words:
            word_text = word_info['text']
            if word_text and word_text not in self._last_words[-5:]:
                self.word_queue.put(word_info)
                self._last_words.append(word_text)

                # Keep last words list bounded
                if len(self._last_words) > 20:
                    self._last_words = self._last_words[-10:]
