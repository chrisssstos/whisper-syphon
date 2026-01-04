"""Audio matching for real-time track and position detection.

Compares live system audio against pre-loaded track audio using
cross-correlation to detect which track is playing and at what position.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of audio matching."""
    track_id: str
    title: str
    artist: str
    position_ms: int
    confidence: float
    deck: int  # 0 = unknown, 1 = deck A, 2 = deck B


class AudioMatcher:
    """Match live audio against pre-loaded tracks using cross-correlation.

    This approach works by:
    1. Pre-loading audio fingerprints (downsampled spectrograms) for each track
    2. Computing fingerprint of live audio
    3. Cross-correlating to find best match and position
    """

    SAMPLE_RATE = 16000
    HOP_LENGTH = 512  # ~32ms per frame at 16kHz
    N_MELS = 32  # Mel bands for fingerprint
    MATCH_WINDOW_SEC = 5.0  # Seconds of audio to match
    MATCH_INTERVAL_SEC = 2.0  # How often to run matching
    MIN_CONFIDENCE = 0.3

    def __init__(self):
        self._tracks: Dict[str, dict] = {}  # track_id -> {title, artist, fingerprint, duration_ms}
        self._lock = threading.Lock()
        self._librosa = None

    def _ensure_librosa(self):
        """Lazily import librosa."""
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
            except ImportError:
                raise ImportError("librosa is required. Install with: pip install librosa")

    def _compute_fingerprint(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute a compact fingerprint from audio using mel spectrogram."""
        self._ensure_librosa()

        # Resample if needed
        if sr != self.SAMPLE_RATE:
            audio = self._librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)

        # Compute mel spectrogram
        mel = self._librosa.feature.melspectrogram(
            y=audio,
            sr=self.SAMPLE_RATE,
            n_mels=self.N_MELS,
            hop_length=self.HOP_LENGTH
        )

        # Convert to log scale
        mel_db = self._librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        return mel_db.astype(np.float32)

    def load_track(self, track_id: str, file_path: str, title: str, artist: str) -> bool:
        """Load and fingerprint a track for matching.

        Args:
            track_id: Unique identifier
            file_path: Path to audio file
            title: Track title
            artist: Track artist

        Returns:
            True if loaded successfully
        """
        self._ensure_librosa()

        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return False

            # Load audio
            audio, sr = self._librosa.load(file_path, sr=self.SAMPLE_RATE, mono=True)
            duration_ms = int(len(audio) / self.SAMPLE_RATE * 1000)

            # Compute fingerprint
            fingerprint = self._compute_fingerprint(audio, self.SAMPLE_RATE)

            with self._lock:
                self._tracks[track_id] = {
                    'title': title,
                    'artist': artist,
                    'fingerprint': fingerprint,
                    'duration_ms': duration_ms
                }

            logger.info(f"Loaded track: {artist} - {title} ({duration_ms}ms, {fingerprint.shape[1]} frames)")
            return True

        except Exception as e:
            logger.error(f"Error loading track {file_path}: {e}")
            return False

    def load_tracks_batch(
        self,
        tracks: List[dict],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        """Load multiple tracks.

        Args:
            tracks: List of dicts with 'id', 'file_path', 'title', 'artist'
            progress_callback: Called with (current, total, track_name)

        Returns:
            Number of tracks loaded successfully
        """
        loaded = 0
        total = len(tracks)

        for i, track in enumerate(tracks):
            if progress_callback:
                progress_callback(i + 1, total, f"{track.get('artist', '?')} - {track.get('title', '?')}")

            file_path = track.get('file_path', '')
            if file_path and Path(file_path).exists():
                if self.load_track(
                    track['id'],
                    file_path,
                    track.get('title', ''),
                    track.get('artist', '')
                ):
                    loaded += 1

        logger.info(f"Loaded {loaded}/{total} tracks for matching")
        return loaded

    def match(self, audio: np.ndarray, sr: int = 16000) -> Optional[MatchResult]:
        """Match audio against loaded tracks.

        Args:
            audio: Audio samples (mono, float32)
            sr: Sample rate

        Returns:
            MatchResult with track info and position, or None if no match
        """
        if len(audio) < sr * 2:  # Need at least 2 seconds
            return None

        with self._lock:
            if not self._tracks:
                return None
            tracks_copy = dict(self._tracks)

        try:
            # Compute fingerprint of live audio
            live_fp = self._compute_fingerprint(audio, sr)

            best_match = None
            best_score = self.MIN_CONFIDENCE
            best_offset = 0

            for track_id, track_data in tracks_copy.items():
                track_fp = track_data['fingerprint']

                # Cross-correlate to find best alignment
                score, offset_frames = self._correlate(live_fp, track_fp)

                if score > best_score:
                    best_score = score
                    best_match = (track_id, track_data)
                    best_offset = offset_frames

            if best_match:
                track_id, track_data = best_match
                # Convert frame offset to milliseconds
                offset_ms = int(best_offset * self.HOP_LENGTH / self.SAMPLE_RATE * 1000)

                return MatchResult(
                    track_id=track_id,
                    title=track_data['title'],
                    artist=track_data['artist'],
                    position_ms=max(0, offset_ms),
                    confidence=best_score,
                    deck=0  # TODO: Detect deck from stereo separation
                )

            return None

        except Exception as e:
            logger.error(f"Match error: {e}")
            return None

    def _correlate(self, live_fp: np.ndarray, track_fp: np.ndarray) -> Tuple[float, int]:
        """Cross-correlate fingerprints to find best match position.

        Args:
            live_fp: Fingerprint of live audio (n_mels, n_frames_live)
            track_fp: Fingerprint of track (n_mels, n_frames_track)

        Returns:
            (score, offset_in_track_frames)
        """
        # Flatten mel bands into single vector per frame, then correlate
        live_flat = live_fp.mean(axis=0)  # Average across mel bands
        track_flat = track_fp.mean(axis=0)

        # Normalized cross-correlation
        live_norm = (live_flat - live_flat.mean()) / (live_flat.std() + 1e-8)
        track_norm = (track_flat - track_flat.mean()) / (track_flat.std() + 1e-8)

        # Full correlation
        correlation = np.correlate(track_norm, live_norm, mode='valid')

        if len(correlation) == 0:
            return 0.0, 0

        # Normalize by length
        correlation = correlation / len(live_norm)

        # Find best match
        best_idx = np.argmax(correlation)
        best_score = correlation[best_idx]

        return float(best_score), int(best_idx)

    def get_loaded_tracks(self) -> List[str]:
        """Get list of loaded track IDs."""
        with self._lock:
            return list(self._tracks.keys())

    def clear(self):
        """Clear all loaded tracks."""
        with self._lock:
            self._tracks.clear()
        logger.info("Cleared all loaded tracks")
