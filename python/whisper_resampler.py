"""
Audio resampler for Whisper integration.
Converts 24kHz audio to 16kHz required by Whisper.
"""

import numpy as np
from scipy.signal import resample_poly


class AudioResampler:
    """Resample audio from 24kHz to 16kHz for Whisper."""

    def __init__(self, input_rate: int = 24000, output_rate: int = 16000):
        self.input_rate = input_rate
        self.output_rate = output_rate
        # 24kHz -> 16kHz = multiply by 2, divide by 3
        self.up = 2
        self.down = 3
        self._residual = np.array([], dtype=np.float32)

    def resample(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Resample audio chunk, handling fractional samples.

        Args:
            audio_chunk: Float32 audio at input_rate (24kHz)

        Returns:
            Resampled audio at output_rate (16kHz)
        """
        # Flatten if needed
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        # Combine with any residual from previous call
        combined = np.concatenate([self._residual, audio_chunk])

        # Need enough samples for polyphase filter
        if len(combined) < 10:
            self._residual = combined
            return np.array([], dtype=np.float32)

        # Resample using polyphase filter
        resampled = resample_poly(combined, self.up, self.down).astype(np.float32)

        # Clear residual after successful resample
        self._residual = np.array([], dtype=np.float32)

        return resampled

    def reset(self):
        """Reset internal state."""
        self._residual = np.array([], dtype=np.float32)
