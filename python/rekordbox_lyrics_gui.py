#!/usr/bin/env python3
"""
Rekordbox Lyrics GUI - Pre-fetch and display synced lyrics alongside Rekordbox DJ sets
"""

import sys
import queue
import threading
import time
import logging
from pathlib import Path
from fractions import Fraction
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFrame, QProgressBar,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QGroupBox,
    QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from typing import Dict, Optional
from PyQt6.QtGui import QFont, QColor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local imports
from lyrics import LRCParser, LyricsFetcher, LyricsCache
from rekordbox import RekordboxMonitor, Track, Playlist
from sync import RkbxOSCReceiver, AudioMatcher, MatchResult, ProLinkListener, DeckStatus, TrackLoaded
from ui import LyricsRenderer
from whisper_resampler import AudioResampler

# ScreenCaptureKit for system audio (macOS 12.3+)
try:
    from screencapture_audio import ScreenCaptureAudio, check_screencapturekit_available
    HAS_SCREENCAPTURE = check_screencapturekit_available()
except ImportError:
    HAS_SCREENCAPTURE = False
    logger.warning("ScreenCaptureKit module not available")

# NDI output
try:
    from cyndilib.sender import Sender
    from cyndilib.video_frame import VideoSendFrame
    from cyndilib.wrapper.ndi_structs import FourCC
    HAS_NDI = True
except ImportError:
    HAS_NDI = False
    logger.warning("cyndilib not available, NDI output disabled")


class NDIOutput:
    """NDI video output sender"""

    def __init__(self, name="Rekordbox Lyrics", width=1920, height=1080, fps=30):
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.sender = None
        self.video_frame = None
        self.running = False

        if not HAS_NDI:
            return

        self.sender = Sender(name)
        self.video_frame = VideoSendFrame()
        self.video_frame.set_resolution(width, height)
        self.video_frame.set_frame_rate(Fraction(fps, 1))
        self.video_frame.set_fourcc(FourCC.RGBA)
        self.sender.set_video_frame(self.video_frame)

    def start(self):
        if self.sender:
            self.sender.open()
            self.running = True
            return True
        return False

    def stop(self):
        if self.sender:
            self.sender.close()
            self.running = False

    def send_frame(self, rgba_array: np.ndarray):
        if not self.sender or not self.running:
            return
        if rgba_array.shape != (self.height, self.width, 4):
            return
        data = np.ascontiguousarray(rgba_array, dtype=np.uint8).ravel()
        self.sender.write_video(data)


class SignalEmitter(QObject):
    """PyQt signal emitter for thread-safe UI updates"""
    status_signal = pyqtSignal(str, str)  # key, value
    progress_signal = pyqtSignal(int, int, str)  # current, total, item_name
    lyrics_signal = pyqtSignal(str)  # current lyrics text
    track_signal = pyqtSignal(str, str)  # title, artist
    error_signal = pyqtSignal(str)  # error message
    analysis_complete = pyqtSignal()


class RekordboxLyricsGUI(QMainWindow):
    """Main GUI for Rekordbox Lyrics Add-on"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rekordbox Lyrics")
        self.setMinimumSize(900, 700)
        self._setup_styles()

        # Components
        self.rekordbox = RekordboxMonitor()
        self.lyrics_fetcher = LyricsFetcher()
        self.lyrics_cache = LyricsCache()
        self.osc_receiver = RkbxOSCReceiver(port=4460)  # rkbx_link default
        self.audio_matcher = AudioMatcher()  # For macOS audio matching fallback
        self.prolink_listener = ProLinkListener()  # Pro DJ Link protocol listener
        self.lyrics_renderer = LyricsRenderer()

        # Track current decks
        self._deck_tracks = {1: None, 2: None}  # deck -> Track object
        self._track_by_rekordbox_id: Dict[int, Track] = {}  # rekordbox ID -> Track

        # Sync mode: "dbpoll" (database polling), "prolink" (Pro DJ Link), or "audio" (audio matching)
        self._sync_mode = "dbpoll"  # Default to database polling (works standalone)
        self._current_match: MatchResult = None

        # NDI
        self.ndi = NDIOutput()

        # Audio capture
        self.screencapture_audio = None
        self.audio_queue = queue.Queue()

        # State
        self.running = False
        self.analyzing = False
        self.current_track = None
        self.current_lyrics = None
        self._pending_track = None  # For thread-safe track handoff from audio worker

        # Signals
        self.signals = SignalEmitter()
        self.signals.status_signal.connect(self._on_status)
        self.signals.progress_signal.connect(self._on_progress)
        self.signals.lyrics_signal.connect(self._on_lyrics)
        self.signals.track_signal.connect(self._on_track_change)
        self.signals.error_signal.connect(self._on_error)
        self.signals.analysis_complete.connect(self._on_analysis_complete)

        # Build UI
        self._setup_ui()

        # Timers
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._render_frame)

        # Try connecting to Rekordbox
        self._connect_rekordbox()

    def _setup_styles(self):
        """Set up the dark theme stylesheet"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: white; }
            QGroupBox {
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #1084d8; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QPushButton#stop_btn {
                background-color: #d83b01;
            }
            QPushButton#stop_btn:hover {
                background-color: #ea4a12;
            }
            QTreeWidget {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #0078d4;
            }
            QTableWidget {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                gridline-color: #444;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: white;
                padding: 5px;
                border: none;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #2d2d2d;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: white;
                padding: 8px 16px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
        """)

    def _setup_ui(self):
        """Build the main UI"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Playlist Analyzer
        self.analyzer_tab = self._create_analyzer_tab()
        self.tabs.addTab(self.analyzer_tab, "Playlist Analyzer")

        # Tab 2: Live Mode
        self.live_tab = self._create_live_tab()
        self.tabs.addTab(self.live_tab, "Live Mode")

        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.status_bar)

    def _create_analyzer_tab(self) -> QWidget:
        """Create the playlist analyzer tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left: Playlist tree
        left_group = QGroupBox("Rekordbox Playlists")
        left_layout = QVBoxLayout(left_group)

        self.playlist_tree = QTreeWidget()
        self.playlist_tree.setHeaderLabel("Playlists")
        self.playlist_tree.itemClicked.connect(self._on_playlist_selected)
        left_layout.addWidget(self.playlist_tree)

        refresh_btn = QPushButton("Refresh Playlists")
        refresh_btn.clicked.connect(self._load_playlists)
        left_layout.addWidget(refresh_btn)

        layout.addWidget(left_group, 1)

        # Right: Analysis controls and status
        right_group = QGroupBox("Analysis")
        right_layout = QVBoxLayout(right_group)

        # Selected playlist info
        self.selected_playlist_label = QLabel("No playlist selected")
        self.selected_playlist_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.selected_playlist_label)

        # Track table
        self.track_table = QTableWidget()
        self.track_table.setColumnCount(3)
        self.track_table.setHorizontalHeaderLabels(["Title", "Artist", "Lyrics"])
        self.track_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.track_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.track_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.track_table.cellDoubleClicked.connect(self._on_track_double_clicked)
        right_layout.addWidget(self.track_table)

        # Progress
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setFormat("%v / %m - %p%")
        right_layout.addWidget(self.analysis_progress)

        self.analysis_status = QLabel("")
        right_layout.addWidget(self.analysis_status)

        # Buttons
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Playlist")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.analyze_btn.setEnabled(False)
        btn_layout.addWidget(self.analyze_btn)

        self.stop_analysis_btn = QPushButton("Stop")
        self.stop_analysis_btn.setObjectName("stop_btn")
        self.stop_analysis_btn.clicked.connect(self._stop_analysis)
        self.stop_analysis_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_analysis_btn)

        right_layout.addLayout(btn_layout)

        layout.addWidget(right_group, 2)

        return tab

    def _create_live_tab(self) -> QWidget:
        """Create the live mode tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Sync mode selector
        sync_group = QGroupBox("Sync Mode")
        sync_layout = QHBoxLayout(sync_group)

        self.sync_mode_group = QButtonGroup()

        self.dbpoll_radio = QRadioButton("Rekordbox History (Recommended)")
        self.dbpoll_radio.setToolTip("Poll Rekordbox database for played tracks. Works standalone, ~1 min delay.")
        self.dbpoll_radio.setChecked(True)
        self.sync_mode_group.addButton(self.dbpoll_radio)
        sync_layout.addWidget(self.dbpoll_radio)

        self.prolink_radio = QRadioButton("Pro DJ Link (CDJs)")
        self.prolink_radio.setToolTip("Real-time sync via network. Requires CDJs or XDJs connected!")
        self.sync_mode_group.addButton(self.prolink_radio)
        sync_layout.addWidget(self.prolink_radio)

        self.audio_radio = QRadioButton("Audio Matching")
        self.audio_radio.setToolTip("Match live audio against pre-analyzed tracks. Less accurate.")
        self.sync_mode_group.addButton(self.audio_radio)
        sync_layout.addWidget(self.audio_radio)

        # Connect mode changes
        self.dbpoll_radio.toggled.connect(lambda checked: setattr(self, '_sync_mode', 'dbpoll') if checked else None)
        self.prolink_radio.toggled.connect(lambda checked: setattr(self, '_sync_mode', 'prolink') if checked else None)
        self.audio_radio.toggled.connect(lambda checked: setattr(self, '_sync_mode', 'audio') if checked else None)

        layout.addWidget(sync_group)

        # Current track display
        track_group = QGroupBox("Now Playing")
        track_layout = QVBoxLayout(track_group)

        self.current_track_label = QLabel("No track detected")
        self.current_track_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.current_track_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        track_layout.addWidget(self.current_track_label)

        self.current_artist_label = QLabel("")
        self.current_artist_label.setStyleSheet("font-size: 14px; color: #888;")
        self.current_artist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        track_layout.addWidget(self.current_artist_label)

        layout.addWidget(track_group)

        # Lyrics preview
        lyrics_group = QGroupBox("Lyrics Preview")
        lyrics_layout = QVBoxLayout(lyrics_group)

        self.lyrics_preview = QTextEdit()
        self.lyrics_preview.setReadOnly(True)
        self.lyrics_preview.setFont(QFont("Helvetica", 14))
        self.lyrics_preview.setMinimumHeight(200)
        lyrics_layout.addWidget(self.lyrics_preview)

        self.current_line_label = QLabel("")
        self.current_line_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #ffcc00; padding: 10px;"
        )
        self.current_line_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_line_label.setWordWrap(True)
        lyrics_layout.addWidget(self.current_line_label)

        layout.addWidget(lyrics_group)

        # Status indicators
        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout(status_group)

        self.ndi_status = QLabel("NDI: Not started")
        self.ndi_status.setStyleSheet("color: #888;")
        status_layout.addWidget(self.ndi_status)

        self.sync_status = QLabel("Sync: Idle")
        self.sync_status.setStyleSheet("color: #888;")
        status_layout.addWidget(self.sync_status)

        self.position_label = QLabel("Position: --:--")
        self.position_label.setStyleSheet("color: #888;")
        status_layout.addWidget(self.position_label)

        layout.addWidget(status_group)

        # Controls
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Live Mode")
        self.start_btn.clicked.connect(self._start_live)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self._stop_live)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        layout.addLayout(controls_layout)

        return tab

    def _connect_rekordbox(self):
        """Try to connect to Rekordbox database"""
        try:
            if self.rekordbox.connect():
                self.signals.status_signal.emit("status", "Connected to Rekordbox")
                self._load_playlists()
            else:
                self.signals.status_signal.emit("status", "Could not connect to Rekordbox")
        except Exception as e:
            logger.error(f"Rekordbox connection error: {e}")
            self.signals.status_signal.emit("status", f"Rekordbox error: {e}")

    def _load_playlists(self):
        """Load playlists from Rekordbox"""
        self.playlist_tree.clear()

        try:
            playlists = self.rekordbox.get_playlists()

            # Build tree structure
            items = {}
            for pl in playlists:
                item = QTreeWidgetItem([pl.name])
                item.setData(0, Qt.ItemDataRole.UserRole, pl.id)

                if pl.parent_id and pl.parent_id in items:
                    items[pl.parent_id].addChild(item)
                else:
                    self.playlist_tree.addTopLevelItem(item)

                items[pl.id] = item

            self.playlist_tree.expandAll()
            self.signals.status_signal.emit("status", f"Loaded {len(playlists)} playlists")

        except Exception as e:
            logger.error(f"Error loading playlists: {e}")
            self.signals.error_signal.emit(f"Error loading playlists: {e}")

    def _on_playlist_selected(self, item: QTreeWidgetItem, column: int):
        """Handle playlist selection"""
        playlist_id = item.data(0, Qt.ItemDataRole.UserRole)
        playlist_name = item.text(0)

        self.selected_playlist_label.setText(playlist_name)
        self.analyze_btn.setEnabled(True)

        # Store selected playlist
        self._selected_playlist_id = playlist_id

        # Load tracks
        self._load_playlist_tracks(playlist_id)

    def _load_playlist_tracks(self, playlist_id: str):
        """Load tracks for the selected playlist"""
        try:
            tracks = self.rekordbox.get_playlist_tracks(playlist_id)
            self._current_tracks = tracks

            self.track_table.setRowCount(len(tracks))

            for i, track in enumerate(tracks):
                self.track_table.setItem(i, 0, QTableWidgetItem(track.title))
                self.track_table.setItem(i, 1, QTableWidgetItem(track.artist))

                # Check cache status
                has_lyrics = self.lyrics_cache.has_lyrics(track.title, track.artist)
                lyrics_status = "Cached" if has_lyrics else "Not fetched"
                self.track_table.setItem(i, 2, QTableWidgetItem(lyrics_status))

            self.signals.status_signal.emit("status", f"Loaded {len(tracks)} tracks - double-click to play lyrics")

        except Exception as e:
            logger.error(f"Error loading tracks: {e}")
            self.signals.error_signal.emit(f"Error loading tracks: {e}")

    def _on_track_double_clicked(self, row: int, column: int):
        """Handle double-click on track to manually start lyrics"""
        if not hasattr(self, '_current_tracks') or row >= len(self._current_tracks):
            return

        track = self._current_tracks[row]
        logger.info(f"Manual track selection: {track.artist} - {track.title}")

        # Switch to Live Mode tab
        self.tabs.setCurrentIndex(1)

        # Start live mode if not running
        if not self.running:
            self._start_live()

        # Load this track's lyrics (start from beginning)
        self._handle_track_detected(track, start_from_beginning=True)

        self.signals.status_signal.emit("status", f"Playing: {track.artist} - {track.title}")

    def _start_analysis(self):
        """Start analyzing the selected playlist"""
        if not hasattr(self, '_current_tracks') or not self._current_tracks:
            return

        self.analyzing = True
        self.analyze_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(True)

        self.analysis_progress.setMaximum(len(self._current_tracks) * 2)  # lyrics + fingerprint
        self.analysis_progress.setValue(0)

        # Start analysis in background thread
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        """Background worker for playlist analysis"""
        tracks = self._current_tracks
        total = len(tracks)

        # Phase 1: Fetch lyrics
        for i, track in enumerate(tracks):
            if not self.analyzing:
                break

            self.signals.progress_signal.emit(i, total * 2, f"Lyrics: {track.artist} - {track.title}")

            if not self.lyrics_cache.has(track.title, track.artist):
                lyrics = self.lyrics_fetcher.search(track.title, track.artist, enhanced=True)
                has_word_ts = lyrics and "<" in lyrics  # Enhanced format has < timestamps
                self.lyrics_cache.put(
                    track.title, track.artist, lyrics,
                    has_word_timestamps=has_word_ts,
                    source="syncedlyrics"
                )

        # Phase 2: Load audio fingerprints for matching
        self.audio_matcher.clear()  # Start fresh
        loaded_count = 0

        for i, track in enumerate(tracks):
            if not self.analyzing:
                break

            self.signals.progress_signal.emit(total + i, total * 2, f"Audio: {track.artist} - {track.title}")

            if track.file_path:
                path = Path(track.file_path)
                if path.exists():
                    track_id = f"{track.title}_{track.artist}"
                    success = self.audio_matcher.load_track(
                        track_id,
                        track.file_path,
                        track.title,
                        track.artist
                    )
                    if success:
                        loaded_count += 1
                        logger.info(f"Loaded audio: {track.artist} - {track.title}")
                    else:
                        logger.warning(f"Failed to load: {track.artist} - {track.title}")
                else:
                    logger.warning(f"File not found: {track.file_path}")
            else:
                logger.warning(f"No path for: {track.artist} - {track.title}")

        logger.info(f"Audio analysis complete: {loaded_count}/{total} tracks loaded")
        self.signals.analysis_complete.emit()

    def _stop_analysis(self):
        """Stop the analysis"""
        self.analyzing = False

    def _on_analysis_complete(self):
        """Handle analysis completion"""
        self.analyzing = False
        self.analyze_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)

        # Refresh track table
        if hasattr(self, '_selected_playlist_id'):
            self._load_playlist_tracks(self._selected_playlist_id)

        self.signals.status_signal.emit("status", "Analysis complete - lyrics cached")

    def _start_live(self):
        """Start live mode"""
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Start NDI
        if self.ndi.start():
            self.ndi_status.setText("NDI: Active")
            self.ndi_status.setStyleSheet("color: #00ff00;")

        if self._sync_mode == "dbpoll":
            self._start_dbpoll_mode()
        elif self._sync_mode == "prolink":
            self._start_prolink_mode()
        else:
            self._start_audio_mode()

        # Start render timer
        self.render_timer.start(33)  # ~30 fps

    def _start_dbpoll_mode(self):
        """Start database polling mode - polls Rekordbox history for track changes."""
        self.sync_status.setText("Sync: Polling Rekordbox history...")
        self.sync_status.setStyleSheet("color: #00ff00;")

        # Start database polling thread
        threading.Thread(target=self._database_poll_worker, daemon=True).start()
        self.signals.status_signal.emit("status", "Live mode started - polling Rekordbox history")

    def _start_prolink_mode(self):
        """Start Pro DJ Link sync mode (network protocol)."""
        # Build track lookup by rekordbox ID from current playlist
        self._track_by_rekordbox_id.clear()
        if hasattr(self, '_current_tracks'):
            for track in self._current_tracks:
                try:
                    # Track.id is the rekordbox database ID
                    rb_id = int(track.id)
                    self._track_by_rekordbox_id[rb_id] = track
                except (ValueError, TypeError):
                    pass
            logger.info(f"Built track lookup with {len(self._track_by_rekordbox_id)} tracks")

        # Set up Pro DJ Link callbacks
        def on_track_loaded(event: TrackLoaded):
            """Called when a track is loaded on a deck."""
            logger.info(f"Pro DJ Link: Track loaded on deck {event.device_number}, ID={event.track_id}")

            # Look up track by rekordbox ID
            track = self._track_by_rekordbox_id.get(event.track_id)
            if track:
                self._deck_tracks[event.device_number] = track
                # Store for main thread and emit signal
                self._pending_track = track
                self.signals.track_signal.emit(track.title, track.artist)
                logger.info(f"Matched track: {track.artist} - {track.title}")
            else:
                # Track not in our analyzed playlist - try all tracks from rekordbox
                all_tracks = self.rekordbox.get_all_tracks()
                for t in all_tracks:
                    try:
                        if int(t.id) == event.track_id:
                            self._deck_tracks[event.device_number] = t
                            self._pending_track = t
                            self.signals.track_signal.emit(t.title, t.artist)
                            logger.info(f"Found track in full collection: {t.artist} - {t.title}")
                            return
                    except (ValueError, TypeError):
                        pass
                logger.warning(f"Track ID {event.track_id} not found in rekordbox database")

        def on_status_update(status: DeckStatus):
            """Called on each status packet from a deck."""
            # Update position for the active deck
            if status.is_playing and status.track_id > 0:
                position_ms = self.prolink_listener.estimate_position_ms(status.device_number)
                if position_ms is not None:
                    self.lyrics_renderer.set_position(position_ms)

                    # Update UI
                    mins = position_ms // 60000
                    secs = (position_ms % 60000) // 1000
                    self.signals.status_signal.emit("position", f"Deck {status.device_number}: {mins:02d}:{secs:02d}")

                    # Get current line for preview
                    current_line = self.lyrics_renderer.get_current_line_text()
                    if current_line:
                        self.signals.lyrics_signal.emit(current_line)

        self.prolink_listener.on_track_loaded = on_track_loaded
        self.prolink_listener.on_status_update = on_status_update

        try:
            self.prolink_listener.start()

            # Check if we're in fallback database polling mode
            if self.prolink_listener.is_database_polling_mode:
                self.sync_status.setText("Sync: Database polling (Rekordbox has port lock)")
                self.sync_status.setStyleSheet("color: #ffaa00;")  # Orange for fallback
                self.signals.status_signal.emit("status", "Using database polling - track detection delayed ~1 min")
                # Start database polling thread
                threading.Thread(target=self._database_poll_worker, daemon=True).start()
            else:
                self.sync_status.setText("Sync: Pro DJ Link active - waiting for Rekordbox")
                self.sync_status.setStyleSheet("color: #00ff00;")
                self.signals.status_signal.emit("status", "Live mode started - Pro DJ Link listening on network")

        except Exception as e:
            logger.error(f"Pro DJ Link error: {e}")
            self.sync_status.setText(f"Sync: Pro DJ Link Error - {e}")
            self.sync_status.setStyleSheet("color: #ff0000;")
            self.signals.error_signal.emit(f"Pro DJ Link failed: {e}")

    def _database_poll_worker(self):
        """Poll rekordbox database for track changes (fallback mode)."""
        last_track_key = None
        POLL_INTERVAL = 2.0  # Poll every 2 seconds

        while self.running:
            try:
                # Get the most recently played track from rekordbox history
                current_track = self.rekordbox.get_current_track()

                if current_track:
                    track_key = f"{current_track.title}|{current_track.artist}"

                    if track_key != last_track_key:
                        last_track_key = track_key
                        logger.info(f"Database poll: New track detected - {current_track.artist} - {current_track.title}")

                        # Update pending track and emit signal
                        self._pending_track = current_track
                        self.signals.track_signal.emit(current_track.title, current_track.artist)

                time.sleep(POLL_INTERVAL)

            except Exception as e:
                logger.error(f"Database poll error: {e}")
                time.sleep(POLL_INTERVAL)

    def _start_osc_mode(self):
        """Start OSC sync mode (rkbx_link for Windows)"""
        # Set up OSC callbacks for rkbx_link
        def on_track_change(deck: int, title: str, artist: str):
            logger.info(f"rkbx_link track change on deck {deck}: {artist} - {title}")
            # Find matching track from analyzed playlist
            if hasattr(self, '_current_tracks'):
                for track in self._current_tracks:
                    # Match by title (case insensitive)
                    if track.title.lower() == title.lower():
                        self._deck_tracks[deck] = track
                        # If this is deck 1 or master, update display
                        self._handle_osc_track_change(deck, track)
                        return
            # Track not found in cache
            logger.warning(f"Track not in cache: {artist} - {title}")

        def on_position_update(deck: int, position_sec: float):
            # Update lyrics renderer with real-time position
            if self._deck_tracks.get(deck):
                position_ms = int(position_sec * 1000)
                self.lyrics_renderer.set_position(position_ms)

        self.osc_receiver.on_track_change = on_track_change
        self.osc_receiver.on_position_update = on_position_update

        try:
            self.osc_receiver.start()
            self.sync_status.setText("Sync: Waiting for rkbx_link on port 4460")
            self.sync_status.setStyleSheet("color: #00ff00;")
        except Exception as e:
            logger.error(f"OSC error: {e}")
            self.sync_status.setText(f"Sync: OSC Error - {e}")
            self.sync_status.setStyleSheet("color: #ff0000;")

        # Start position update thread for UI updates
        threading.Thread(target=self._osc_position_worker, daemon=True).start()
        self.signals.status_signal.emit("status", "Live mode started - run rkbx_link with Rekordbox")

    def _start_audio_mode(self):
        """Start audio matching mode (macOS)"""
        if not HAS_SCREENCAPTURE:
            self.sync_status.setText("Sync: ScreenCaptureKit not available")
            self.sync_status.setStyleSheet("color: #ff0000;")
            self.signals.error_signal.emit("ScreenCaptureKit not available on this system")
            return

        # Check if audio matcher has tracks loaded
        loaded_tracks = self.audio_matcher.get_loaded_tracks()
        if not loaded_tracks:
            self.sync_status.setText("Sync: No audio loaded - Analyze Playlist first!")
            self.sync_status.setStyleSheet("color: #ff0000;")
            self.signals.error_signal.emit("Run 'Analyze Playlist' first to load audio fingerprints")
            return

        logger.info(f"Audio matcher has {len(loaded_tracks)} tracks loaded")

        # Start audio capture
        try:
            self.screencapture_audio = ScreenCaptureAudio()
            self.screencapture_audio.start()
            self.sync_status.setText("Sync: Audio matching active")
            self.sync_status.setStyleSheet("color: #00ff00;")
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self.sync_status.setText(f"Sync: Audio Error - {e}")
            self.sync_status.setStyleSheet("color: #ff0000;")
            return

        # Start audio matching worker
        threading.Thread(target=self._audio_match_worker, daemon=True).start()
        self.signals.status_signal.emit("status", "Live mode started - audio matching active")

    def _audio_match_worker(self):
        """Background worker for audio matching (macOS)"""
        audio_buffer = []
        SAMPLE_RATE = 16000
        MATCH_WINDOW_SEC = 5.0
        MATCH_INTERVAL_SEC = 2.0
        samples_needed = int(SAMPLE_RATE * MATCH_WINDOW_SEC)

        last_match_time = 0
        last_position_ms = 0
        position_update_time = time.time()

        while self.running and self.screencapture_audio:
            try:
                # Get audio from capture
                audio_chunk = self.screencapture_audio.read(timeout=0.1)
                if audio_chunk is not None:
                    audio_buffer.extend(audio_chunk)

                    # Keep buffer at max size
                    if len(audio_buffer) > samples_needed * 2:
                        audio_buffer = audio_buffer[-samples_needed:]

                # Run matching periodically
                now = time.time()
                if now - last_match_time >= MATCH_INTERVAL_SEC and len(audio_buffer) >= samples_needed:
                    last_match_time = now

                    # Match audio
                    audio_array = np.array(audio_buffer[-samples_needed:], dtype=np.float32)
                    match = self.audio_matcher.match(audio_array, SAMPLE_RATE)

                    if match and match.confidence > 0.3:
                        self._current_match = match
                        last_position_ms = match.position_ms
                        position_update_time = time.time()

                        # Find track object and emit signal (don't update UI directly from thread!)
                        if hasattr(self, '_current_tracks'):
                            for track in self._current_tracks:
                                track_id = f"{track.title}_{track.artist}"
                                if track_id == match.track_id:
                                    # Check if track changed - emit signal for main thread
                                    if self.current_track is None or self.current_track.title != track.title:
                                        self.current_track = track
                                        # Use signal to update UI on main thread
                                        self.signals.track_signal.emit(track.title, track.artist)
                                        # Store track for lyrics loading (will be handled by signal)
                                        self._pending_track = track
                                    break

                        logger.debug(f"Match: {match.track_id} at {match.position_ms}ms (conf: {match.confidence:.2f})")

                # Interpolate position between matches
                if self._current_match:
                    elapsed = time.time() - position_update_time
                    interpolated_ms = last_position_ms + int(elapsed * 1000)
                    self.lyrics_renderer.set_position(interpolated_ms)

                    # Update UI
                    mins = interpolated_ms // 60000
                    secs = (interpolated_ms % 60000) // 1000
                    self.signals.status_signal.emit("position", f"Position: {mins:02d}:{secs:02d}")

                    # Get current line for preview
                    current_line = self.lyrics_renderer.get_current_line_text()
                    if current_line:
                        self.signals.lyrics_signal.emit(current_line)

                time.sleep(0.02)  # 50Hz loop

            except Exception as e:
                logger.error(f"Audio match worker error: {e}")
                time.sleep(0.1)

    def _osc_position_worker(self):
        """Background worker for OSC position UI updates"""
        while self.running:
            try:
                # Get position from OSC receiver for deck 1 (or active deck)
                for deck in [1, 2]:
                    if self._deck_tracks.get(deck):
                        position_ms = self.osc_receiver.get_deck_position_ms(deck)

                        # Update UI
                        mins = position_ms // 60000
                        secs = (position_ms % 60000) // 1000
                        self.signals.status_signal.emit("position", f"Deck {deck}: {mins:02d}:{secs:02d}")

                        # Get current line for preview
                        current_line = self.lyrics_renderer.get_current_line_text()
                        if current_line:
                            self.signals.lyrics_signal.emit(current_line)

                        break  # Only show one deck for now

                time.sleep(0.05)  # 20Hz updates

            except Exception as e:
                logger.error(f"OSC position worker error: {e}")
                time.sleep(0.1)

    def _handle_osc_track_change(self, deck: int, track: Track):
        """Handle track change from rkbx_link OSC"""
        self.signals.track_signal.emit(track.title, track.artist)

        # Load lyrics
        lyrics_lrc = self.lyrics_cache.get(track.title, track.artist)
        if lyrics_lrc:
            parsed = LRCParser.parse(lyrics_lrc)
            self.lyrics_renderer.set_lyrics(parsed)
            self.current_lyrics = parsed

            # Show in preview
            plain_text = "\n".join(line.raw_text for line in parsed)
            self.lyrics_preview.setPlainText(plain_text)

            self.sync_status.setText(f"Sync: Deck {deck} - {track.title[:20]}")
            logger.info(f"Loaded lyrics for deck {deck}: {track.artist} - {track.title}")
        else:
            self.lyrics_renderer.clear()
            self.current_lyrics = None
            self.lyrics_preview.setPlainText("No lyrics cached for this track")
            self.sync_status.setText(f"Sync: Deck {deck} - No lyrics")

    def _handle_track_detected(self, track: Track, start_from_beginning: bool = False):
        """Handle track detection from Rekordbox monitoring or manual selection"""
        self.signals.track_signal.emit(track.title, track.artist)

        # Load lyrics
        lyrics_lrc = self.lyrics_cache.get(track.title, track.artist)
        if lyrics_lrc:
            parsed = LRCParser.parse(lyrics_lrc)
            self.lyrics_renderer.set_lyrics(parsed)
            self.current_lyrics = parsed

            # Show in preview
            plain_text = "\n".join(line.raw_text for line in parsed)
            self.lyrics_preview.setPlainText(plain_text)

            # Start playback timer
            if start_from_beginning:
                self._playback_offset_ms = 0
            else:
                # Rekordbox marks tracks ~1 min in
                self._playback_offset_ms = 60000

            self._playback_start_time = time.time()
            self._is_playing = True

            logger.info(f"Loaded lyrics for {track.artist} - {track.title}, starting at {self._playback_offset_ms}ms")
        else:
            self.lyrics_renderer.clear()
            self.current_lyrics = None
            self.lyrics_preview.setPlainText("No lyrics cached - analyze playlist first")
            self._is_playing = False

    def _render_frame(self):
        """Render and send NDI frame"""
        if not self.running:
            return

        frame = self.lyrics_renderer.render()
        self.ndi.send_frame(frame)

    def _stop_live(self):
        """Stop live mode"""
        self.running = False

        # Stop Pro DJ Link listener
        self.prolink_listener.stop()

        # Stop OSC receiver
        self.osc_receiver.stop()

        # Stop audio capture
        if self.screencapture_audio:
            try:
                self.screencapture_audio.stop()
            except:
                pass
            self.screencapture_audio = None

        # Clear current match but keep loaded tracks
        self._current_match = None

        # Stop NDI
        self.ndi.stop()
        self.ndi_status.setText("NDI: Stopped")
        self.ndi_status.setStyleSheet("color: #888;")

        # Stop render timer
        self.render_timer.stop()

        # Clear deck tracking
        self._deck_tracks = {1: None, 2: None}

        self.sync_status.setText("Sync: Idle")
        self.sync_status.setStyleSheet("color: #888;")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.signals.status_signal.emit("status", "Live mode stopped")

    # Signal handlers
    def _on_status(self, key: str, value: str):
        if key == "status":
            self.status_bar.setText(value)
        elif key == "position":
            self.position_label.setText(value)

    def _on_progress(self, current: int, total: int, item: str):
        self.analysis_progress.setValue(current)
        self.analysis_status.setText(item)

    def _on_lyrics(self, text: str):
        self.current_line_label.setText(text)

    def _on_track_change(self, title: str, artist: str):
        """Handle track change signal - runs on main thread"""
        self.current_track_label.setText(title)
        self.current_artist_label.setText(artist)

        # Check if there's a pending track from audio matching that needs lyrics loaded
        if hasattr(self, '_pending_track') and self._pending_track is not None:
            track = self._pending_track
            self._pending_track = None  # Clear it

            # Load lyrics (safe to do Qt operations here - we're on main thread)
            lyrics_lrc = self.lyrics_cache.get(track.title, track.artist)
            if lyrics_lrc:
                parsed = LRCParser.parse(lyrics_lrc)
                self.lyrics_renderer.set_lyrics(parsed)
                self.current_lyrics = parsed

                # Show in preview (Qt widget - must be on main thread)
                plain_text = "\n".join(line.raw_text for line in parsed)
                self.lyrics_preview.setPlainText(plain_text)

                self.sync_status.setText(f"Sync: {track.title[:25]}...")
                logger.info(f"Loaded lyrics for: {track.artist} - {track.title}")
            else:
                self.lyrics_renderer.clear()
                self.current_lyrics = None
                self.lyrics_preview.setPlainText("No lyrics cached for this track")
                self.sync_status.setText(f"Sync: No lyrics - {track.title[:20]}")

    def _on_error(self, message: str):
        self.status_bar.setText(f"Error: {message}")
        self.status_bar.setStyleSheet("color: #ff0000; padding: 5px;")

    def closeEvent(self, event):
        """Handle window close"""
        self._stop_live()
        self._stop_analysis()
        self.rekordbox.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = RekordboxLyricsGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
