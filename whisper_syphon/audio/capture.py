"""Audio capture from input device."""

import sounddevice as sd
import numpy as np
from queue import Queue
from typing import Optional, Callable
import threading


class AudioCapture:
    """Captures audio from an input device and feeds it to a queue."""

    SAMPLE_RATE = 16000  # Whisper native sample rate
    CHANNELS = 1
    BLOCK_SIZE = 1600  # 100ms at 16kHz

    def __init__(self, device_index: int, audio_queue: Queue):
        """Initialize audio capture.

        Args:
            device_index: Index of the input device to capture from.
            audio_queue: Queue to push audio chunks to.
        """
        self.device_index = device_index
        self.audio_queue = audio_queue
        self.stream: Optional[sd.InputStream] = None
        self.running = False
        self._lock = threading.Lock()

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")

        # Copy audio data and push to queue
        audio_chunk = indata[:, 0].copy()  # Mono
        self.audio_queue.put(audio_chunk)

    def start(self) -> None:
        """Start audio capture."""
        with self._lock:
            if self.running:
                return

            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype='float32',
                callback=self._audio_callback,
                blocksize=self.BLOCK_SIZE
            )
            self.stream.start()
            self.running = True
            print(f"Audio capture started (device {self.device_index})")

    def stop(self) -> None:
        """Stop audio capture."""
        with self._lock:
            if not self.running:
                return

            self.running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            print("Audio capture stopped")

    def is_running(self) -> bool:
        """Check if capture is running."""
        return self.running
