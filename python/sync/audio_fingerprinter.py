"""Audio fingerprinting for track identification and position detection.

Uses chromaprint for generating fingerprints and a custom matcher
for detecting playback position within pre-analyzed tracks.
"""

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Dict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FingerprintMatch:
    """Result of a fingerprint match."""
    track_id: str
    offset_ms: int
    confidence: float  # 0.0 to 1.0


class AudioFingerprinter:
    """Audio fingerprinting for track identification and position detection.

    Pre-analyzes tracks to create fingerprint databases, then matches
    live audio to determine which track is playing and at what position.
    """

    # Fingerprint parameters
    SAMPLE_RATE = 16000  # Chromaprint expects mono audio
    CHUNK_DURATION_MS = 3000  # Size of fingerprint chunks for position matching
    CHUNK_OVERLAP_MS = 1000  # Overlap between chunks

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the fingerprinter.

        Args:
            db_path: Path to SQLite database for storing fingerprints.
                    Default: ~/.whisper-syphon/fingerprints.db
        """
        if db_path is None:
            db_path = Path.home() / ".whisper-syphon" / "fingerprints.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._chromaprint = None
        self._init_db()

        # In-memory cache for fast matching
        self._fingerprint_cache: Dict[str, List[Tuple[int, bytes]]] = {}

    def _ensure_chromaprint(self):
        """Lazily import chromaprint."""
        if self._chromaprint is None:
            try:
                import chromaprint
                self._chromaprint = chromaprint
            except ImportError:
                raise ImportError(
                    "pyacoustid/chromaprint is required. Install with: pip install pyacoustid"
                )

    def _init_db(self):
        """Initialize the fingerprint database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    artist TEXT,
                    duration_ms INTEGER,
                    fingerprinted_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id TEXT NOT NULL,
                    offset_ms INTEGER NOT NULL,
                    fingerprint BLOB NOT NULL,
                    FOREIGN KEY (track_id) REFERENCES tracks(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fp_track
                ON fingerprints(track_id)
            """)

            conn.commit()

    def _load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load an audio file and convert to mono float32.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Convert to float32
            audio = audio.astype(np.float32)

            return audio, sr

        except ImportError:
            # Fallback to librosa
            try:
                import librosa
                audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, mono=True)
                return audio.astype(np.float32), sr
            except ImportError:
                raise ImportError(
                    "soundfile or librosa is required for audio loading. "
                    "Install with: pip install soundfile"
                )

    def fingerprint_track(
        self,
        track_id: str,
        audio_path: str,
        title: str = "",
        artist: str = ""
    ) -> bool:
        """Fingerprint a track and store in the database.

        Creates multiple fingerprints at different positions in the track
        to enable position detection during live playback.

        Args:
            track_id: Unique identifier for the track
            audio_path: Path to audio file
            title: Track title (for reference)
            artist: Track artist (for reference)

        Returns:
            True if fingerprinting successful
        """
        self._ensure_chromaprint()

        try:
            # Load audio
            audio, sr = self._load_audio_file(audio_path)
            duration_ms = int(len(audio) / sr * 1000)

            logger.info(f"Fingerprinting: {artist} - {title} ({duration_ms}ms)")

            # Generate fingerprint for the whole track
            fp = self._chromaprint.fingerprint(audio, sr)

            # Store track info
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tracks (id, title, artist, duration_ms)
                    VALUES (?, ?, ?, ?)
                """, (track_id, title, artist, duration_ms))

                # Delete old fingerprints
                conn.execute("DELETE FROM fingerprints WHERE track_id = ?", (track_id,))

                # Create fingerprints at regular intervals
                chunk_samples = int(self.CHUNK_DURATION_MS * sr / 1000)
                overlap_samples = int(self.CHUNK_OVERLAP_MS * sr / 1000)
                step_samples = chunk_samples - overlap_samples

                offset_ms = 0
                pos = 0

                while pos + chunk_samples <= len(audio):
                    chunk = audio[pos:pos + chunk_samples]
                    chunk_fp = self._chromaprint.fingerprint(chunk, sr)

                    if chunk_fp:
                        # Store as binary
                        fp_bytes = chunk_fp.encode('utf-8')
                        conn.execute("""
                            INSERT INTO fingerprints (track_id, offset_ms, fingerprint)
                            VALUES (?, ?, ?)
                        """, (track_id, offset_ms, fp_bytes))

                    pos += step_samples
                    offset_ms = int(pos / sr * 1000)

                conn.commit()

            logger.info(f"Fingerprinted {artist} - {title}: {offset_ms}ms worth of chunks")
            return True

        except Exception as e:
            logger.error(f"Error fingerprinting {audio_path}: {e}")
            return False

    def fingerprint_batch(
        self,
        tracks: List[Dict],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, bool]:
        """Fingerprint multiple tracks.

        Args:
            tracks: List of dicts with 'id', 'file_path', 'title', 'artist' keys
            progress_callback: Called with (current, total, track_name)

        Returns:
            Dict mapping track_id to success status
        """
        results = {}
        total = len(tracks)

        for i, track in enumerate(tracks):
            track_id = track.get('id', '')
            file_path = track.get('file_path', '')
            title = track.get('title', '')
            artist = track.get('artist', '')

            if progress_callback:
                progress_callback(i + 1, total, f"{artist} - {title}")

            if not file_path or not Path(file_path).exists():
                logger.warning(f"Skipping {title}: file not found")
                results[track_id] = False
                continue

            results[track_id] = self.fingerprint_track(
                track_id, file_path, title, artist
            )

        return results

    def load_fingerprints_to_memory(self, track_ids: Optional[List[str]] = None):
        """Load fingerprints into memory for faster matching.

        Args:
            track_ids: List of track IDs to load, or None for all
        """
        self._fingerprint_cache.clear()

        with sqlite3.connect(self.db_path) as conn:
            if track_ids:
                placeholders = ','.join('?' * len(track_ids))
                cursor = conn.execute(f"""
                    SELECT track_id, offset_ms, fingerprint
                    FROM fingerprints
                    WHERE track_id IN ({placeholders})
                    ORDER BY track_id, offset_ms
                """, track_ids)
            else:
                cursor = conn.execute("""
                    SELECT track_id, offset_ms, fingerprint
                    FROM fingerprints
                    ORDER BY track_id, offset_ms
                """)

            for row in cursor:
                track_id, offset_ms, fp_bytes = row
                if track_id not in self._fingerprint_cache:
                    self._fingerprint_cache[track_id] = []
                self._fingerprint_cache[track_id].append((offset_ms, fp_bytes))

        logger.info(f"Loaded {len(self._fingerprint_cache)} tracks into memory")

    def match_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        expected_track_id: Optional[str] = None
    ) -> Optional[FingerprintMatch]:
        """Match an audio chunk against stored fingerprints.

        Args:
            audio: Audio samples (mono, float32)
            sample_rate: Sample rate of the audio
            expected_track_id: If set, only match against this track (faster)

        Returns:
            FingerprintMatch with track_id and offset, or None if no match
        """
        self._ensure_chromaprint()

        if len(audio) < sample_rate * 2:  # Need at least 2 seconds
            logger.debug("Audio chunk too short for matching")
            return None

        try:
            # Generate fingerprint for the query audio
            query_fp = self._chromaprint.fingerprint(audio, sample_rate)
            if not query_fp:
                return None

            query_bytes = query_fp.encode('utf-8')

            # Search in memory cache first
            if self._fingerprint_cache:
                return self._match_in_memory(query_bytes, expected_track_id)

            # Fall back to database
            return self._match_in_db(query_bytes, expected_track_id)

        except Exception as e:
            logger.error(f"Error matching audio: {e}")
            return None

    def _match_in_memory(
        self,
        query_fp: bytes,
        expected_track_id: Optional[str]
    ) -> Optional[FingerprintMatch]:
        """Match against in-memory fingerprints."""
        best_match = None
        best_score = 0.0

        tracks_to_search = (
            {expected_track_id: self._fingerprint_cache.get(expected_track_id, [])}
            if expected_track_id and expected_track_id in self._fingerprint_cache
            else self._fingerprint_cache
        )

        for track_id, fingerprints in tracks_to_search.items():
            for offset_ms, stored_fp in fingerprints:
                score = self._compare_fingerprints(query_fp, stored_fp)
                if score > best_score and score > 0.3:  # Minimum threshold
                    best_score = score
                    best_match = FingerprintMatch(
                        track_id=track_id,
                        offset_ms=offset_ms,
                        confidence=score
                    )

        return best_match

    def _match_in_db(
        self,
        query_fp: bytes,
        expected_track_id: Optional[str]
    ) -> Optional[FingerprintMatch]:
        """Match against database fingerprints."""
        best_match = None
        best_score = 0.0

        with sqlite3.connect(self.db_path) as conn:
            if expected_track_id:
                cursor = conn.execute("""
                    SELECT track_id, offset_ms, fingerprint
                    FROM fingerprints
                    WHERE track_id = ?
                """, (expected_track_id,))
            else:
                cursor = conn.execute("""
                    SELECT track_id, offset_ms, fingerprint
                    FROM fingerprints
                """)

            for row in cursor:
                track_id, offset_ms, stored_fp = row
                score = self._compare_fingerprints(query_fp, stored_fp)
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = FingerprintMatch(
                        track_id=track_id,
                        offset_ms=offset_ms,
                        confidence=score
                    )

        return best_match

    def _compare_fingerprints(self, fp1: bytes, fp2: bytes) -> float:
        """Compare two fingerprints and return similarity score.

        Uses hash comparison for efficiency.

        Args:
            fp1: First fingerprint bytes
            fp2: Second fingerprint bytes

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple hash-based comparison
        # In production, would use proper chromaprint comparison
        if fp1 == fp2:
            return 1.0

        # Compute Jaccard-like similarity on fingerprint chunks
        try:
            # Split fingerprints into chunks and compare
            chunk_size = 20
            chunks1 = set(fp1[i:i+chunk_size] for i in range(0, len(fp1)-chunk_size, chunk_size))
            chunks2 = set(fp2[i:i+chunk_size] for i in range(0, len(fp2)-chunk_size, chunk_size))

            if not chunks1 or not chunks2:
                return 0.0

            intersection = len(chunks1 & chunks2)
            union = len(chunks1 | chunks2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def is_track_fingerprinted(self, track_id: str) -> bool:
        """Check if a track has been fingerprinted.

        Args:
            track_id: Track identifier

        Returns:
            True if fingerprints exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM tracks WHERE id = ?",
                (track_id,)
            )
            return cursor.fetchone() is not None

    def get_stats(self) -> Dict[str, int]:
        """Get fingerprint database statistics.

        Returns:
            Dict with track and fingerprint counts
        """
        with sqlite3.connect(self.db_path) as conn:
            tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            fingerprints = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]

        return {
            'tracks': tracks,
            'fingerprints': fingerprints,
            'cached_tracks': len(self._fingerprint_cache)
        }

    def clear(self):
        """Clear all fingerprints."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM fingerprints")
            conn.execute("DELETE FROM tracks")
            conn.commit()
        self._fingerprint_cache.clear()
        logger.info("Fingerprint database cleared")
