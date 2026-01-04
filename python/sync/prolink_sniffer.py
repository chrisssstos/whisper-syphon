"""
Pro DJ Link packet sniffer using scapy.

This approach captures packets without needing to bind to ports,
which is necessary when Rekordbox has exclusive locks on the ports.

Note: May require running with sudo on macOS for packet capture.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List

logger = logging.getLogger(__name__)

# Try to import scapy
try:
    from scapy.all import sniff, UDP, IP, Raw, conf
    # Disable scapy warnings
    conf.verb = 0
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    logger.warning("scapy not available - Pro DJ Link sniffer disabled")

# Pro DJ Link constants
PROLINK_MAGIC = b'Qspt1WmJOL'
ANNOUNCE_PORT = 50000
STATUS_PORT = 50001


@dataclass
class DeckStatus:
    """Status of a single deck/player."""
    device_number: int
    track_id: int
    bpm: float
    pitch_percent: float
    beat: int
    bar_beat: int
    play_state: int
    is_playing: bool
    is_master: bool
    is_on_air: bool
    beat_count: int
    packet_time: float


@dataclass
class TrackLoaded:
    """Event when a track is loaded on a deck."""
    device_number: int
    track_id: int
    slot: int


class ProLinkSniffer:
    """
    Sniffs Pro DJ Link packets using scapy.

    This works even when Rekordbox has exclusive port bindings.
    Requires elevated privileges (sudo) on most systems.
    """

    def __init__(self, interface: str = None):
        """
        Initialize the sniffer.

        Args:
            interface: Network interface to sniff on (e.g., 'en0').
                      If None, will try to auto-detect.
        """
        self.interface = interface
        self._running = False
        self._sniff_thread: Optional[threading.Thread] = None

        # State
        self._deck_status: Dict[int, DeckStatus] = {}
        self._last_track_id: Dict[int, int] = {}

        # Callbacks
        self.on_track_loaded: Optional[Callable[[TrackLoaded], None]] = None
        self.on_status_update: Optional[Callable[[DeckStatus], None]] = None

    def start(self):
        """Start sniffing for Pro DJ Link packets."""
        if not HAS_SCAPY:
            raise ImportError("scapy is required for packet sniffing")

        if self._running:
            return

        self._running = True
        self._sniff_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self._sniff_thread.start()
        logger.info(f"Pro DJ Link sniffer started on interface {self.interface or 'default'}")

    def stop(self):
        """Stop sniffing."""
        self._running = False
        if self._sniff_thread:
            self._sniff_thread.join(timeout=2.0)
            self._sniff_thread = None
        logger.info("Pro DJ Link sniffer stopped")

    def _sniff_loop(self):
        """Main sniffing loop."""
        # Build BPF filter for Pro DJ Link ports
        bpf_filter = f"udp port {ANNOUNCE_PORT} or udp port {STATUS_PORT}"

        try:
            # Sniff packets with a timeout so we can check _running flag
            while self._running:
                packets = sniff(
                    iface=self.interface,
                    filter=bpf_filter,
                    count=10,  # Process in batches
                    timeout=1.0,
                    store=True
                )

                for pkt in packets:
                    if not self._running:
                        break
                    self._process_packet(pkt)

        except PermissionError:
            logger.error("Permission denied - try running with sudo for packet capture")
        except Exception as e:
            logger.error(f"Sniffer error: {e}")

    def _process_packet(self, pkt):
        """Process a captured packet."""
        try:
            if not pkt.haslayer(UDP) or not pkt.haslayer(Raw):
                return

            data = bytes(pkt[Raw].load)

            # Check for Pro DJ Link magic header
            if not data.startswith(PROLINK_MAGIC):
                return

            dst_port = pkt[UDP].dport

            if dst_port == STATUS_PORT and len(data) >= 0xa0:
                self._parse_cdj_status(data)
            elif dst_port == ANNOUNCE_PORT:
                # Device announcement - just log for now
                if len(data) > 30:
                    device_name = data[11:31].decode('ascii', errors='ignore').rstrip('\x00')
                    logger.debug(f"Device announcement: {device_name}")

        except Exception as e:
            logger.debug(f"Error processing packet: {e}")

    def _parse_cdj_status(self, data: bytes):
        """Parse a CDJ status packet."""
        import struct

        try:
            device_number = data[0x21]
            track_slot = data[0x29]
            track_id = struct.unpack('>I', data[0x2c:0x30])[0]
            play_state = data[0x7b]
            flags = data[0x89]
            is_playing = bool(flags & 0x40)
            is_master = bool(flags & 0x20)
            is_on_air = bool(flags & 0x08)
            bpm_raw = struct.unpack('>H', data[0x92:0x94])[0]
            bpm = bpm_raw / 100.0 if bpm_raw > 0 else 0.0
            pitch_raw = struct.unpack('>I', data[0x98:0x9c])[0]
            pitch_percent = ((pitch_raw / 0x100000) - 1.0) * 100.0
            beat_count = struct.unpack('>I', data[0xa0:0xa4])[0] if len(data) > 0xa3 else 0
            bar_beat = data[0xa6] if len(data) > 0xa6 else 1

            status = DeckStatus(
                device_number=device_number,
                track_id=track_id,
                bpm=bpm,
                pitch_percent=pitch_percent,
                beat=bar_beat,
                bar_beat=bar_beat,
                play_state=play_state,
                is_playing=is_playing,
                is_master=is_master,
                is_on_air=is_on_air,
                beat_count=beat_count,
                packet_time=time.time()
            )

            self._deck_status[device_number] = status

            # Check for track change
            last_track = self._last_track_id.get(device_number, 0)
            if track_id != last_track and track_id > 0:
                self._last_track_id[device_number] = track_id
                if self.on_track_loaded:
                    event = TrackLoaded(
                        device_number=device_number,
                        track_id=track_id,
                        slot=track_slot
                    )
                    self.on_track_loaded(event)
                logger.info(f"Track loaded on deck {device_number}: ID={track_id}")

            if self.on_status_update:
                self.on_status_update(status)

        except Exception as e:
            logger.error(f"Error parsing CDJ status: {e}")

    def get_deck_status(self, deck: int) -> Optional[DeckStatus]:
        """Get current status of a deck."""
        return self._deck_status.get(deck)

    def estimate_position_ms(self, deck: int) -> Optional[int]:
        """Estimate playback position in milliseconds."""
        status = self._deck_status.get(deck)
        if not status or status.bpm <= 0:
            return None
        ms_per_beat = 60000.0 / status.bpm
        elapsed = time.time() - status.packet_time
        if status.is_playing:
            adjusted_beats = status.beat_count + (elapsed * status.bpm / 60.0)
        else:
            adjusted_beats = status.beat_count
        return int(adjusted_beats * ms_per_beat)


def test_sniffer():
    """Test the sniffer - run with sudo."""
    logging.basicConfig(level=logging.DEBUG)

    sniffer = ProLinkSniffer(interface='en0')

    def on_track(event):
        print(f"Track: deck={event.device_number}, id={event.track_id}")

    def on_status(status):
        if status.is_playing:
            print(f"Deck {status.device_number}: bpm={status.bpm:.1f}, beat={status.beat_count}")

    sniffer.on_track_loaded = on_track
    sniffer.on_status_update = on_status

    try:
        sniffer.start()
        print("Sniffing... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()


if __name__ == "__main__":
    test_sniffer()
