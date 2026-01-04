"""Fetch synced lyrics from online sources."""

import logging
from typing import Optional, List, Callable, Dict, Any

logger = logging.getLogger(__name__)


class LyricsFetcher:
    """Fetch synced lyrics from online providers.

    Uses syncedlyrics library to fetch from:
    - Musixmatch (best for word-level timestamps)
    - LRCLIB (free, community-maintained)
    - NetEase (Chinese songs)
    """

    def __init__(self, providers: Optional[List[str]] = None):
        """Initialize the lyrics fetcher.

        Args:
            providers: List of provider names to use (order matters).
                      Default: ["Musixmatch", "Lrclib", "NetEase"]
        """
        self.providers = providers or ["Musixmatch", "Lrclib", "NetEase"]
        self._syncedlyrics = None

    def _ensure_syncedlyrics(self):
        """Lazily import syncedlyrics to avoid import errors if not installed."""
        if self._syncedlyrics is None:
            try:
                import syncedlyrics
                self._syncedlyrics = syncedlyrics
            except ImportError:
                raise ImportError(
                    "syncedlyrics is required. Install with: pip install syncedlyrics"
                )

    def search(
        self,
        title: str,
        artist: str,
        enhanced: bool = True,
        duration_ms: Optional[int] = None
    ) -> Optional[str]:
        """Search for synced lyrics for a track.

        Args:
            title: Track title
            artist: Track artist
            enhanced: If True, prefer word-level timestamps when available
            duration_ms: Optional track duration for better matching

        Returns:
            LRC format lyrics string, or None if not found
        """
        self._ensure_syncedlyrics()

        search_query = f"{title} {artist}"
        logger.info(f"Searching lyrics for: {search_query}")

        try:
            # syncedlyrics.search returns LRC string or None
            lyrics = self._syncedlyrics.search(
                search_query,
                enhanced=enhanced
            )

            if lyrics:
                logger.info(f"Found lyrics for: {search_query}")
                return lyrics
            else:
                logger.warning(f"No lyrics found for: {search_query}")
                return None

        except Exception as e:
            logger.error(f"Error fetching lyrics for {search_query}: {e}")
            return None

    def search_batch(
        self,
        tracks: List[Dict[str, Any]],
        enhanced: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Optional[str]]:
        """Batch fetch lyrics for multiple tracks.

        Args:
            tracks: List of dicts with 'title' and 'artist' keys
            enhanced: If True, prefer word-level timestamps
            progress_callback: Called with (current, total, track_name) for progress updates

        Returns:
            Dict mapping "title|artist" to LRC content (None if not found)
        """
        results = {}
        total = len(tracks)

        for i, track in enumerate(tracks):
            title = track.get('title', '')
            artist = track.get('artist', '')
            key = f"{title}|{artist}"

            if progress_callback:
                progress_callback(i + 1, total, f"{artist} - {title}")

            lyrics = self.search(title, artist, enhanced=enhanced)
            results[key] = lyrics

        return results
