"""Monitor Rekordbox database for playlists and track changes."""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a track from Rekordbox."""
    id: str
    title: str
    artist: str
    file_path: Optional[str] = None
    duration_ms: int = 0
    bpm: float = 0.0
    key: str = ""
    album: str = ""


@dataclass
class Playlist:
    """Represents a playlist from Rekordbox."""
    id: str
    name: str
    parent_id: Optional[str] = None
    track_count: int = 0
    is_folder: bool = False


class RekordboxMonitor:
    """Monitor Rekordbox database for playlists and track changes.

    Uses pyrekordbox to read from the Rekordbox database.
    Note: Rekordbox must not be running in some cases due to database locking.
    """

    def __init__(self, poll_interval: float = 2.0):
        """Initialize the Rekordbox monitor.

        Args:
            poll_interval: How often to poll for track changes (seconds)
        """
        self.poll_interval = poll_interval
        self._db = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_history_id: Optional[str] = None

        # Callbacks
        self.on_track_change: Optional[Callable[[Track], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def _ensure_db(self):
        """Lazily connect to Rekordbox database."""
        if self._db is None:
            try:
                from pyrekordbox import Rekordbox6Database
                self._db = Rekordbox6Database()
                logger.info("Connected to Rekordbox database")
            except ImportError:
                raise ImportError(
                    "pyrekordbox is required. Install with: pip install pyrekordbox"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Rekordbox database: {e}")
                raise

    def connect(self) -> bool:
        """Connect to the Rekordbox database.

        Returns:
            True if connection successful
        """
        try:
            self._ensure_db()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def get_playlists(self) -> List[Playlist]:
        """Get all playlists from Rekordbox.

        Returns:
            List of Playlist objects
        """
        self._ensure_db()
        playlists = []

        try:
            for pl in self._db.get_playlist():
                playlists.append(Playlist(
                    id=str(pl.ID),
                    name=pl.Name or "Unnamed",
                    parent_id=str(pl.ParentID) if pl.ParentID else None,
                    track_count=len(list(pl.Songs)) if hasattr(pl, 'Songs') else 0,
                    is_folder=pl.Attribute == 1 if hasattr(pl, 'Attribute') else False
                ))
        except Exception as e:
            logger.error(f"Error getting playlists: {e}")

        return playlists

    def get_playlist_tracks(self, playlist_id: str) -> List[Track]:
        """Get all tracks in a playlist.

        Args:
            playlist_id: The playlist ID

        Returns:
            List of Track objects
        """
        self._ensure_db()
        tracks = []

        try:
            playlist = None
            for pl in self._db.get_playlist():
                if str(pl.ID) == playlist_id:
                    playlist = pl
                    break

            if playlist is None:
                logger.warning(f"Playlist {playlist_id} not found")
                return []

            for song in playlist.Songs:
                content = song.Content
                if content:
                    tracks.append(self._content_to_track(content))

        except Exception as e:
            logger.error(f"Error getting playlist tracks: {e}")

        return tracks

    def get_all_tracks(self) -> List[Track]:
        """Get all tracks from the Rekordbox collection.

        Returns:
            List of Track objects
        """
        self._ensure_db()
        tracks = []

        try:
            for content in self._db.get_content():
                tracks.append(self._content_to_track(content))
        except Exception as e:
            logger.error(f"Error getting all tracks: {e}")

        return tracks

    def _content_to_track(self, content: Any) -> Track:
        """Convert a pyrekordbox content object to a Track.

        Args:
            content: pyrekordbox DjmdContent object

        Returns:
            Track object
        """
        artist_name = ""
        if hasattr(content, 'Artist') and content.Artist:
            artist_name = content.Artist.Name or ""

        album_name = ""
        if hasattr(content, 'Album') and content.Album:
            album_name = content.Album.Name or ""

        file_path = None
        from urllib.parse import unquote

        # In Rekordbox 6/7, FolderPath often contains the FULL path, not just folder
        # Check FolderPath first - it may be the complete path
        if hasattr(content, 'FolderPath') and content.FolderPath:
            raw_path = str(content.FolderPath)
            # Handle file:// URLs
            if raw_path.startswith('file://localhost'):
                raw_path = raw_path.replace('file://localhost', '')
            elif raw_path.startswith('file://'):
                raw_path = raw_path.replace('file://', '')
            file_path = unquote(raw_path)

        # If FolderPath looks like a directory (no extension), try adding FileName/FileNameL
        if file_path and not Path(file_path).suffix:
            filename = None
            if hasattr(content, 'FileName') and content.FileName:
                filename = content.FileName
            elif hasattr(content, 'FileNameL') and content.FileNameL:
                filename = content.FileNameL
            if filename:
                file_path = str(Path(file_path) / filename)

        # Log for debugging
        title = content.Title or "Unknown"
        if file_path:
            exists = Path(file_path).exists()
            logger.debug(f"Track '{title}' path: {file_path} (exists: {exists})")
        else:
            logger.debug(f"Track '{title}' has no file path")

        return Track(
            id=str(content.ID),
            title=content.Title or "Unknown",
            artist=artist_name,
            file_path=file_path,
            duration_ms=int(content.Length * 1000) if hasattr(content, 'Length') and content.Length else 0,
            bpm=float(content.BPM) if hasattr(content, 'BPM') and content.BPM else 0.0,
            key=content.Key.ScaleName if hasattr(content, 'Key') and content.Key else "",
            album=album_name
        )

    def get_recently_played(self, limit: int = 10) -> List[Track]:
        """Get recently played tracks from history.

        Args:
            limit: Maximum number of tracks to return

        Returns:
            List of Track objects, most recent first
        """
        self._ensure_db()
        tracks = []

        try:
            # Get history entries sorted by date
            history = list(self._db.get_history())
            # Sort by creation date descending
            history.sort(key=lambda h: h.DateCreated if hasattr(h, 'DateCreated') else "", reverse=True)

            for entry in history[:limit]:
                if hasattr(entry, 'Content') and entry.Content:
                    tracks.append(self._content_to_track(entry.Content))

        except Exception as e:
            logger.error(f"Error getting play history: {e}")

        return tracks

    def get_current_track(self) -> Optional[Track]:
        """Get the most recently played track.

        Note: Rekordbox marks tracks as "played" about 1 minute into playback.

        Returns:
            Track object or None
        """
        tracks = self.get_recently_played(limit=1)
        return tracks[0] if tracks else None

    def start_monitoring(self):
        """Start monitoring for track changes.

        The on_track_change callback will be called when a new track is detected.
        """
        if self._monitoring:
            logger.warning("Already monitoring")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started Rekordbox monitoring")

    def stop_monitoring(self):
        """Stop monitoring for track changes."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Stopped Rekordbox monitoring")

    def _monitor_loop(self):
        """Background loop that polls for track changes."""
        while self._monitoring:
            try:
                current = self.get_current_track()

                if current:
                    current_id = f"{current.title}|{current.artist}"

                    if current_id != self._last_history_id:
                        self._last_history_id = current_id
                        logger.info(f"Track change detected: {current.artist} - {current.title}")

                        if self.on_track_change:
                            self.on_track_change(current)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                if self.on_error:
                    self.on_error(e)

            time.sleep(self.poll_interval)

    def close(self):
        """Close the database connection and stop monitoring."""
        self.stop_monitoring()
        self._db = None
        logger.info("Closed Rekordbox connection")
