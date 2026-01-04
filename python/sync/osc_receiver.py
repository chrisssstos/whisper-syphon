"""OSC receiver for rkbx_link integration.

Receives real-time track and position data from rkbx_link via OSC.
rkbx_link reads Rekordbox memory directly for instant sync.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict
from pythonosc import dispatcher, osc_server

logger = logging.getLogger(__name__)


@dataclass
class DeckState:
    """State of a single deck."""
    title: str = ""
    artist: str = ""
    album: str = ""
    position_sec: float = 0.0
    bpm: float = 0.0
    beat: float = 0.0
    is_master: bool = False


@dataclass
class RkbxState:
    """Full state from rkbx_link."""
    decks: Dict[int, DeckState] = field(default_factory=lambda: {1: DeckState(), 2: DeckState()})
    master_deck: int = 1

    def get_master(self) -> DeckState:
        """Get the current master deck state."""
        return self.decks.get(self.master_deck, DeckState())


class RkbxOSCReceiver:
    """Receive real-time data from rkbx_link via OSC.

    rkbx_link sends OSC messages with track info and position.
    Default port is 4460 (rkbx_link destination).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4460):
        """Initialize the OSC receiver.

        Args:
            host: Host to listen on
            port: Port to listen on (rkbx_link default destination is 4460)
        """
        self.host = host
        self.port = port
        self._server: Optional[osc_server.ThreadingOSCUDPServer] = None
        self._state = RkbxState()
        self._lock = threading.Lock()

        # Callbacks
        self.on_track_change: Optional[Callable[[int, str, str], None]] = None  # deck, title, artist
        self.on_position_update: Optional[Callable[[int, float], None]] = None  # deck, position_sec

        # Track change detection
        self._last_track: Dict[int, str] = {}

    def _get_deck_num(self, address: str) -> Optional[int]:
        """Extract deck number from OSC address.

        Args:
            address: OSC address like /track/1/title or /track/master/title

        Returns:
            Deck number (1-4) or None for master
        """
        parts = address.split('/')
        if len(parts) >= 3:
            deck_part = parts[2]
            if deck_part == 'master':
                return None  # Will use master_deck
            try:
                return int(deck_part)
            except ValueError:
                pass
        return None

    def _handle_time(self, address: str, time_sec: float):
        """Handle /time/[deck] message."""
        deck = self._get_deck_num(address)
        if deck is None:
            deck = self._state.master_deck

        with self._lock:
            if deck in self._state.decks:
                self._state.decks[deck].position_sec = time_sec

        if self.on_position_update:
            self.on_position_update(deck, time_sec)

    def _handle_title(self, address: str, title: str):
        """Handle /track/[deck]/title message."""
        deck = self._get_deck_num(address)
        if deck is None:
            deck = self._state.master_deck

        with self._lock:
            if deck in self._state.decks:
                old_title = self._state.decks[deck].title
                self._state.decks[deck].title = title

                # Check for track change
                track_key = f"{deck}:{title}"
                if track_key != self._last_track.get(deck):
                    self._last_track[deck] = track_key
                    artist = self._state.decks[deck].artist

                    if self.on_track_change and title:
                        logger.info(f"Track change on deck {deck}: {artist} - {title}")
                        self.on_track_change(deck, title, artist)

    def _handle_artist(self, address: str, artist: str):
        """Handle /track/[deck]/artist message."""
        deck = self._get_deck_num(address)
        if deck is None:
            deck = self._state.master_deck

        with self._lock:
            if deck in self._state.decks:
                self._state.decks[deck].artist = artist

    def _handle_album(self, address: str, album: str):
        """Handle /track/[deck]/album message."""
        deck = self._get_deck_num(address)
        if deck is None:
            deck = self._state.master_deck

        with self._lock:
            if deck in self._state.decks:
                self._state.decks[deck].album = album

    def _handle_bpm(self, address: str, bpm: float):
        """Handle /bpm/[deck]/current message."""
        deck = self._get_deck_num(address)
        if deck is None:
            deck = self._state.master_deck

        with self._lock:
            if deck in self._state.decks:
                self._state.decks[deck].bpm = bpm

    def _handle_beat(self, address: str, beat: float):
        """Handle /beat/[deck] message."""
        deck = self._get_deck_num(address)
        if deck is None:
            deck = self._state.master_deck

        with self._lock:
            if deck in self._state.decks:
                self._state.decks[deck].beat = beat

    def start(self):
        """Start the OSC server."""
        disp = dispatcher.Dispatcher()

        # Register handlers for all deck variants
        for deck in ['master', '1', '2', '3', '4']:
            disp.map(f"/time/{deck}", self._handle_time)
            disp.map(f"/track/{deck}/title", self._handle_title)
            disp.map(f"/track/{deck}/artist", self._handle_artist)
            disp.map(f"/track/{deck}/album", self._handle_album)
            disp.map(f"/bpm/{deck}/current", self._handle_bpm)
            disp.map(f"/beat/{deck}", self._handle_beat)

        try:
            self._server = osc_server.ThreadingOSCUDPServer(
                (self.host, self.port), disp
            )
            logger.info(f"OSC server listening on {self.host}:{self.port}")

            # Start server in background thread
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()

        except Exception as e:
            logger.error(f"Failed to start OSC server: {e}")
            raise

    def stop(self):
        """Stop the OSC server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("OSC server stopped")

    def get_state(self) -> RkbxState:
        """Get current state snapshot."""
        with self._lock:
            return RkbxState(
                decks={k: DeckState(
                    title=v.title,
                    artist=v.artist,
                    album=v.album,
                    position_sec=v.position_sec,
                    bpm=v.bpm,
                    beat=v.beat,
                    is_master=(k == self._state.master_deck)
                ) for k, v in self._state.decks.items()},
                master_deck=self._state.master_deck
            )

    def get_deck_position_ms(self, deck: int) -> int:
        """Get position for a deck in milliseconds."""
        with self._lock:
            if deck in self._state.decks:
                return int(self._state.decks[deck].position_sec * 1000)
            return 0

    def get_deck_track(self, deck: int) -> tuple:
        """Get (title, artist) for a deck."""
        with self._lock:
            if deck in self._state.decks:
                d = self._state.decks[deck]
                return (d.title, d.artist)
            return ("", "")
