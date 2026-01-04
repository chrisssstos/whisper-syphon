"""Playback synchronization using audio fingerprinting.

Matches live audio to pre-fingerprinted tracks to determine
the current playback position with high accuracy.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, List
import numpy as np

from .audio_fingerprinter import AudioFingerprinter, FingerprintMatch

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """Current synchronization state."""
    track_id: Optional[str] = None
    position_ms: int = 0
    confidence: float = 0.0
    last_match_time: float = 0.0
    is_playing: bool = False


class PlaybackSync:
    """Synchronize lyrics display with audio playback.

    Uses audio fingerprinting to detect track and position, then
    interpolates position between matches for smooth sync.
    """

    # Sync parameters
    MATCH_INTERVAL_SECONDS = 10.0  # How often to re-match audio
    MATCH_AUDIO_SECONDS = 5.0  # Amount of audio to use for matching
    MIN_CONFIDENCE = 0.4  # Minimum match confidence to accept
    POSITION_SMOOTHING = 0.8  # Smoothing factor for position interpolation

    def __init__(
        self,
        fingerprinter: AudioFingerprinter,
        sample_rate: int = 16000
    ):
        """Initialize playback sync.

        Args:
            fingerprinter: AudioFingerprinter instance with pre-analyzed tracks
            sample_rate: Expected sample rate of incoming audio
        """
        self.fingerprinter = fingerprinter
        self.sample_rate = sample_rate

        # State
        self._state = SyncState()
        self._audio_buffer: List[np.ndarray] = []
        self._buffer_lock = threading.Lock()

        # Timing
        self._last_match_time = 0.0
        self._last_position_update = 0.0
        self._playback_start_time = 0.0

        # Callbacks
        self.on_track_change: Optional[Callable[[str, int], None]] = None
        self.on_position_update: Optional[Callable[[int], None]] = None

    @property
    def current_track_id(self) -> Optional[str]:
        """Get the currently detected track ID."""
        return self._state.track_id

    @property
    def current_position_ms(self) -> int:
        """Get the current interpolated playback position."""
        if not self._state.is_playing:
            return self._state.position_ms

        # Interpolate position based on time since last match
        elapsed = time.time() - self._last_position_update
        interpolated = self._state.position_ms + int(elapsed * 1000)

        return interpolated

    def set_expected_track(self, track_id: str):
        """Set the expected track for faster matching.

        Called when Rekordbox reports a track change (with delay),
        to narrow down the fingerprint search.

        Args:
            track_id: Expected track identifier
        """
        logger.info(f"Expected track set: {track_id}")
        # Don't immediately switch - wait for audio confirmation
        self._expected_track_id = track_id

    def insert_audio(self, audio: np.ndarray):
        """Insert audio samples for matching.

        Args:
            audio: Audio samples (mono, float32, at self.sample_rate)
        """
        with self._buffer_lock:
            self._audio_buffer.append(audio.copy())

            # Keep buffer at manageable size (last N seconds)
            max_samples = int(self.MATCH_AUDIO_SECONDS * 2 * self.sample_rate)
            total_samples = sum(len(chunk) for chunk in self._audio_buffer)

            while total_samples > max_samples and len(self._audio_buffer) > 1:
                removed = self._audio_buffer.pop(0)
                total_samples -= len(removed)

    def process_step(self) -> Optional[int]:
        """Process a sync step and potentially trigger a match.

        Should be called regularly (e.g., every frame or audio block).

        Returns:
            Current position in milliseconds, or None if not synced
        """
        current_time = time.time()

        # Check if we should re-match
        if current_time - self._last_match_time >= self.MATCH_INTERVAL_SECONDS:
            self._attempt_match()
            self._last_match_time = current_time

        # Update interpolated position
        if self._state.is_playing:
            self._last_position_update = current_time
            position = self.current_position_ms

            if self.on_position_update:
                self.on_position_update(position)

            return position

        return None

    def _attempt_match(self):
        """Attempt to match buffered audio against fingerprints."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return

            # Concatenate buffer
            audio = np.concatenate(self._audio_buffer)

        # Need enough audio for matching
        min_samples = int(self.MATCH_AUDIO_SECONDS * self.sample_rate)
        if len(audio) < min_samples:
            logger.debug(f"Not enough audio for matching: {len(audio)} < {min_samples}")
            return

        # Use the most recent portion
        audio = audio[-min_samples:]

        # Attempt match
        expected = getattr(self, '_expected_track_id', None)
        match = self.fingerprinter.match_audio(
            audio,
            self.sample_rate,
            expected_track_id=expected if self._state.track_id else None
        )

        if match and match.confidence >= self.MIN_CONFIDENCE:
            self._handle_match(match)
        else:
            logger.debug(f"No confident match found (confidence threshold: {self.MIN_CONFIDENCE})")

    def _handle_match(self, match: FingerprintMatch):
        """Handle a successful fingerprint match.

        Args:
            match: FingerprintMatch result
        """
        track_changed = match.track_id != self._state.track_id

        if track_changed:
            logger.info(f"Track detected: {match.track_id} at {match.offset_ms}ms "
                       f"(confidence: {match.confidence:.2f})")

            old_track = self._state.track_id
            self._state.track_id = match.track_id
            self._state.is_playing = True

            if self.on_track_change:
                self.on_track_change(match.track_id, match.offset_ms)
        else:
            logger.debug(f"Position update: {match.offset_ms}ms "
                        f"(confidence: {match.confidence:.2f})")

        # Update position with smoothing
        if self._state.is_playing:
            # Smooth the position to avoid jumps
            current_interpolated = self.current_position_ms
            diff = abs(match.offset_ms - current_interpolated)

            if diff < 2000:  # Within 2 seconds - smooth it
                self._state.position_ms = int(
                    self.POSITION_SMOOTHING * current_interpolated +
                    (1 - self.POSITION_SMOOTHING) * match.offset_ms
                )
            else:
                # Large difference - snap to new position
                self._state.position_ms = match.offset_ms
        else:
            self._state.position_ms = match.offset_ms
            self._state.is_playing = True

        self._state.confidence = match.confidence
        self._state.last_match_time = time.time()
        self._last_position_update = time.time()

    def reset(self):
        """Reset sync state (e.g., when stopping playback)."""
        self._state = SyncState()
        with self._buffer_lock:
            self._audio_buffer.clear()
        self._last_match_time = 0.0
        self._expected_track_id = None
        logger.info("Playback sync reset")

    def pause(self):
        """Pause position interpolation."""
        self._state.is_playing = False
        logger.debug("Playback sync paused")

    def resume(self):
        """Resume position interpolation."""
        self._state.is_playing = True
        self._last_position_update = time.time()
        logger.debug("Playback sync resumed")

    def get_state(self) -> SyncState:
        """Get current sync state.

        Returns:
            Copy of current SyncState
        """
        return SyncState(
            track_id=self._state.track_id,
            position_ms=self.current_position_ms,
            confidence=self._state.confidence,
            last_match_time=self._state.last_match_time,
            is_playing=self._state.is_playing
        )
