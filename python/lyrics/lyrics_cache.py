"""SQLite cache for synced lyrics."""

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LyricsCache:
    """SQLite-based cache for storing fetched lyrics.

    Lyrics are keyed by normalized hash of title + artist.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the lyrics cache.

        Args:
            db_path: Path to SQLite database file.
                    Default: ~/.whisper-syphon/lyrics_cache.db
        """
        if db_path is None:
            db_path = Path.home() / ".whisper-syphon" / "lyrics_cache.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lyrics (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    lrc_content TEXT,
                    has_word_timestamps INTEGER DEFAULT 0,
                    source TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_title_artist
                ON lyrics(title, artist)
            """)

            conn.commit()

    @staticmethod
    def _make_key(title: str, artist: str) -> str:
        """Create a unique key from title and artist.

        Args:
            title: Track title
            artist: Track artist

        Returns:
            SHA256 hash of normalized title + artist
        """
        # Normalize: lowercase, strip whitespace
        normalized = f"{title.lower().strip()}|{artist.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, title: str, artist: str) -> Optional[str]:
        """Retrieve cached lyrics for a track.

        Args:
            title: Track title
            artist: Track artist

        Returns:
            LRC content string, or None if not cached
        """
        key = self._make_key(title, artist)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT lrc_content FROM lyrics WHERE id = ?",
                (key,)
            )
            row = cursor.fetchone()

        if row and row[0]:
            logger.debug(f"Cache hit for: {artist} - {title}")
            return row[0]

        logger.debug(f"Cache miss for: {artist} - {title}")
        return None

    def put(
        self,
        title: str,
        artist: str,
        lrc_content: Optional[str],
        has_word_timestamps: bool = False,
        source: Optional[str] = None
    ):
        """Store lyrics in the cache.

        Args:
            title: Track title
            artist: Track artist
            lrc_content: LRC format lyrics (None to mark as not found)
            has_word_timestamps: Whether lyrics have word-level timing
            source: Provider that supplied the lyrics
        """
        key = self._make_key(title, artist)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lyrics
                (id, title, artist, lrc_content, has_word_timestamps, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                key,
                title,
                artist,
                lrc_content,
                1 if has_word_timestamps else 0,
                source,
                now
            ))
            conn.commit()

        logger.debug(f"Cached lyrics for: {artist} - {title}")

    def has(self, title: str, artist: str) -> bool:
        """Check if lyrics exist in cache (even if empty/not found).

        Args:
            title: Track title
            artist: Track artist

        Returns:
            True if entry exists in cache
        """
        key = self._make_key(title, artist)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM lyrics WHERE id = ?",
                (key,)
            )
            return cursor.fetchone() is not None

    def has_lyrics(self, title: str, artist: str) -> bool:
        """Check if we have actual lyrics content (not just a not-found entry).

        Args:
            title: Track title
            artist: Track artist

        Returns:
            True if lyrics content exists
        """
        key = self._make_key(title, artist)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT lrc_content FROM lyrics WHERE id = ? AND lrc_content IS NOT NULL",
                (key,)
            )
            row = cursor.fetchone()
            return row is not None and row[0] is not None and len(row[0]) > 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with 'total', 'with_lyrics', 'not_found' counts
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM lyrics").fetchone()[0]
            with_lyrics = conn.execute(
                "SELECT COUNT(*) FROM lyrics WHERE lrc_content IS NOT NULL AND lrc_content != ''"
            ).fetchone()[0]
            with_word_ts = conn.execute(
                "SELECT COUNT(*) FROM lyrics WHERE has_word_timestamps = 1"
            ).fetchone()[0]

        return {
            'total': total,
            'with_lyrics': with_lyrics,
            'not_found': total - with_lyrics,
            'with_word_timestamps': with_word_ts
        }

    def clear(self):
        """Clear all cached lyrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM lyrics")
            conn.commit()
        logger.info("Lyrics cache cleared")

    def delete(self, title: str, artist: str):
        """Delete a specific cache entry.

        Args:
            title: Track title
            artist: Track artist
        """
        key = self._make_key(title, artist)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM lyrics WHERE id = ?", (key,))
            conn.commit()
