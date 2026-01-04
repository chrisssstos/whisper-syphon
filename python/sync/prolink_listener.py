"""
Pro DJ Link protocol listener for Rekordbox integration.

Listens for CDJ status packets on the network to get real-time track info,
playback position, and BPM from Rekordbox.

Based on the DJ Link Ecosystem Analysis by Deep Symmetry:
https://djl-analysis.deepsymmetry.org/djl-analysis/

Ports used:
- UDP 50000: Device announcements (keep-alive)
- UDP 50001: Beat/status packets (track info, position)
"""

import socket
import struct
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List
import netifaces

logger = logging.getLogger(__name__)

# Pro DJ Link magic header (all packets start with this)
PROLINK_MAGIC = b'Qspt1WmJOL'

# Ports
ANNOUNCE_PORT = 50000
STATUS_PORT = 50001

# Packet types (byte after magic header)
PACKET_TYPE_CDJ_ANNOUNCE = 0x06  # CDJ keep-alive
PACKET_TYPE_MIXER_ANNOUNCE = 0x06  # Mixer keep-alive
PACKET_TYPE_CDJ_STATUS = 0x0a  # CDJ status update
PACKET_TYPE_BEAT = 0x28  # Beat packet

# Play states (from byte 0x7b in status packet)
PLAY_STATE_EMPTY = 0x00
PLAY_STATE_LOADING = 0x02
PLAY_STATE_PLAYING = 0x03
PLAY_STATE_LOOPING = 0x04
PLAY_STATE_PAUSED = 0x05
PLAY_STATE_CUED = 0x06
PLAY_STATE_CUEING = 0x07
PLAY_STATE_SEARCHING = 0x09
PLAY_STATE_ENDED = 0x11


@dataclass
class DeckStatus:
    """Status of a single deck/player."""
    device_number: int  # 1-4
    track_id: int  # Rekordbox database ID
    bpm: float  # Current BPM (accounting for pitch)
    pitch_percent: float  # Pitch adjustment (-100 to +100)
    beat: int  # Current beat (1-4)
    bar_beat: int  # Beat within bar
    play_state: int  # Play state code
    is_playing: bool
    is_master: bool
    is_on_air: bool
    # Position tracking
    beat_count: int  # Total beats from start
    packet_time: float  # When this status was received


@dataclass
class TrackLoaded:
    """Event when a track is loaded on a deck."""
    device_number: int
    track_id: int  # Rekordbox database ID
    slot: int  # 0=no track, 1=CD, 2=SD, 3=USB, 4=rekordbox


class ProLinkListener:
    """
    Listens for Pro DJ Link network packets from Rekordbox.

    Provides real-time track info, playback position, and BPM
    without requiring audio fingerprinting.
    """

    def __init__(self, device_name: str = "LyricsSync", device_number: int = 5):
        """
        Initialize the Pro DJ Link listener.

        Args:
            device_name: Name to announce ourselves as (max 20 chars)
            device_number: Virtual device number (use 5-6 to avoid conflicts)
        """
        self.device_name = device_name[:20].ljust(20, '\x00')
        self.device_number = device_number

        self._running = False
        self._announce_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None

        # Sockets
        self._announce_socket: Optional[socket.socket] = None
        self._status_socket: Optional[socket.socket] = None

        # Track state per deck
        self._deck_status: Dict[int, DeckStatus] = {}
        self._last_track_id: Dict[int, int] = {}  # deck -> last track_id

        # Callbacks
        self.on_track_loaded: Optional[Callable[[TrackLoaded], None]] = None
        self.on_status_update: Optional[Callable[[DeckStatus], None]] = None
        self.on_beat: Optional[Callable[[int, int], None]] = None  # deck, beat

        # Network info
        self._local_ip: Optional[str] = None
        self._mac_address: Optional[bytes] = None
        self._get_network_info()

        # Fallback mode when ports unavailable
        self._use_database_polling = False

    def _get_network_info(self):
        """Get local IP and MAC address for announcements."""
        try:
            # Find the first non-loopback interface with an IPv4 address
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)

                # Get IPv4 address
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr.get('addr', '')
                        if ip and not ip.startswith('127.'):
                            self._local_ip = ip

                            # Get MAC address
                            if netifaces.AF_LINK in addrs:
                                mac_str = addrs[netifaces.AF_LINK][0].get('addr', '')
                                if mac_str:
                                    self._mac_address = bytes.fromhex(mac_str.replace(':', ''))

                            logger.info(f"Using network interface {iface}: IP={ip}")
                            return

            logger.warning("Could not find suitable network interface")
        except Exception as e:
            logger.error(f"Error getting network info: {e}")

    def start(self):
        """Start listening for Pro DJ Link packets."""
        if self._running:
            return

        self._running = True

        # Create and bind sockets
        # Rekordbox binds exclusively to ports 50000/50001, so we need SO_REUSEPORT
        # which must be set BEFORE bind() on macOS
        try:
            # Announce socket (send keep-alives, receive announcements)
            self._announce_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Must set REUSEPORT before REUSEADDR on some systems
            if hasattr(socket, 'SO_REUSEPORT'):
                self._announce_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self._announce_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._announce_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            # Try binding - if it fails, we'll use an alternative port for sending only
            try:
                self._announce_socket.bind(('', ANNOUNCE_PORT))
            except OSError:
                # Can't bind to 50000, use alternative port for sending announcements
                self._announce_socket.bind(('', 0))  # Let OS pick a port
                logger.warning(f"Could not bind to port {ANNOUNCE_PORT}, using ephemeral port for announcements")
            self._announce_socket.settimeout(1.0)

            # Status socket - this is where we receive CDJ status
            self._status_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if hasattr(socket, 'SO_REUSEPORT'):
                self._status_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self._status_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            try:
                self._status_socket.bind(('', STATUS_PORT))
            except OSError:
                # Can't bind to 50001 - Rekordbox has exclusive lock
                logger.warning(f"Could not bind to port {STATUS_PORT} - Rekordbox has exclusive lock")
                logger.warning("Pro DJ Link direct listening unavailable")
                logger.warning("Falling back to rekordbox database polling mode")

                # Close sockets and signal fallback mode
                self._status_socket.close()
                self._status_socket = None
                self._use_database_polling = True

            if self._status_socket:
                self._status_socket.settimeout(1.0)

        except OSError as e:
            if not self._use_database_polling:
                logger.error(f"Failed to bind sockets: {e}")
                self._running = False
                raise

        # Start threads (only if we have sockets)
        if self._announce_socket:
            self._announce_thread = threading.Thread(target=self._announce_loop, daemon=True)
            self._announce_thread.start()

        if self._status_socket:
            self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
            self._status_thread.start()
            logger.info(f"Pro DJ Link listener started on ports {ANNOUNCE_PORT}/{STATUS_PORT}")
        elif self._use_database_polling:
            logger.info("Pro DJ Link using database polling fallback mode")

    def stop(self):
        """Stop listening."""
        self._running = False

        if self._announce_socket:
            try:
                self._announce_socket.close()
            except:
                pass
            self._announce_socket = None

        if self._status_socket:
            try:
                self._status_socket.close()
            except:
                pass
            self._status_socket = None

        if self._announce_thread:
            self._announce_thread.join(timeout=2.0)
            self._announce_thread = None

        if self._status_thread:
            self._status_thread.join(timeout=2.0)
            self._status_thread = None

        logger.info("Pro DJ Link listener stopped")

    def _announce_loop(self):
        """Send keep-alive announcements and receive device announcements."""
        last_announce = 0
        ANNOUNCE_INTERVAL = 1.5  # Send every 1.5 seconds

        while self._running:
            try:
                # Send our keep-alive announcement
                now = time.time()
                if now - last_announce >= ANNOUNCE_INTERVAL:
                    self._send_keepalive()
                    last_announce = now

                # Receive announcements from other devices
                try:
                    data, addr = self._announce_socket.recvfrom(1024)
                    if data.startswith(PROLINK_MAGIC):
                        self._handle_announce_packet(data, addr)
                except socket.timeout:
                    pass

            except Exception as e:
                if self._running:
                    logger.error(f"Announce loop error: {e}")
                time.sleep(0.1)

    def _status_loop(self):
        """Receive and process status packets."""
        while self._running:
            try:
                data, addr = self._status_socket.recvfrom(1024)
                if data.startswith(PROLINK_MAGIC):
                    self._handle_status_packet(data, addr)
            except socket.timeout:
                pass
            except Exception as e:
                if self._running:
                    logger.error(f"Status loop error: {e}")
                time.sleep(0.01)

    def _send_keepalive(self):
        """Send a keep-alive packet to announce our presence."""
        if not self._local_ip or not self._mac_address:
            return

        # Build keep-alive packet
        # Format based on CDJ keep-alive structure
        packet = bytearray()
        packet.extend(PROLINK_MAGIC)  # Magic header
        packet.append(0x06)  # Packet type: keep-alive
        packet.extend(self.device_name.encode('ascii')[:20])  # Device name (20 bytes)
        packet.append(0x01)  # Unknown
        packet.append(0x02)  # Device type (2 = virtual CDJ)
        packet.extend(self._mac_address)  # MAC address (6 bytes)

        # IP address (4 bytes)
        ip_parts = [int(x) for x in self._local_ip.split('.')]
        packet.extend(bytes(ip_parts))

        packet.append(self.device_number)  # Device number
        packet.append(0x00)  # Unknown
        packet.append(0x00)  # Unknown
        packet.append(0x01)  # Unknown (active flag?)

        # Pad to standard size
        while len(packet) < 54:
            packet.append(0x00)

        try:
            # Broadcast on the local network
            self._announce_socket.sendto(bytes(packet), ('<broadcast>', ANNOUNCE_PORT))
        except Exception as e:
            logger.debug(f"Failed to send keepalive: {e}")

    def _handle_announce_packet(self, data: bytes, addr):
        """Handle an incoming device announcement packet."""
        if len(data) < 32:
            return

        packet_type = data[10]

        if packet_type == 0x06:  # Keep-alive
            # Extract device info
            device_name = data[11:31].decode('ascii', errors='ignore').rstrip('\x00')
            device_number = data[36] if len(data) > 36 else 0

            # Log at INFO level so user can see devices
            logger.info(f"Pro DJ Link device found: {device_name} (#{device_number}) from {addr[0]}")

    def _handle_status_packet(self, data: bytes, addr):
        """Handle an incoming CDJ status packet."""
        if len(data) < 0xa0:  # Need at least 160 bytes for full status
            return

        packet_type = data[10]

        if packet_type == 0x0a:  # CDJ status
            self._parse_cdj_status(data)
        elif packet_type == 0x28:  # Beat packet
            self._parse_beat_packet(data)

    def _parse_cdj_status(self, data: bytes):
        """Parse a CDJ status packet."""
        try:
            # Device number (byte 0x21 = 33)
            device_number = data[0x21]

            # Track source/slot info (bytes 0x28-0x29)
            track_device = data[0x28]  # Device track was loaded from
            track_slot = data[0x29]  # Slot (1=CD, 2=SD, 3=USB, 4=rekordbox)

            # Rekordbox track ID (bytes 0x2c-0x2f, big-endian)
            track_id = struct.unpack('>I', data[0x2c:0x30])[0]

            # Play state (byte 0x7b = 123)
            play_state = data[0x7b]

            # Flags byte (byte 0x89 = 137)
            flags = data[0x89]
            is_playing = bool(flags & 0x40)  # Bit 6
            is_master = bool(flags & 0x20)  # Bit 5
            is_on_air = bool(flags & 0x08)  # Bit 3

            # BPM (bytes 0x92-0x93, big-endian, divide by 100)
            bpm_raw = struct.unpack('>H', data[0x92:0x94])[0]
            bpm = bpm_raw / 100.0 if bpm_raw > 0 else 0.0

            # Pitch (bytes 0x8c-0x8f or 0x98-0x9b)
            # 0x00100000 = no adjustment (0%)
            pitch_raw = struct.unpack('>I', data[0x98:0x9c])[0]
            pitch_percent = ((pitch_raw / 0x100000) - 1.0) * 100.0

            # Beat counter (bytes 0xa0-0xa3)
            beat_count = struct.unpack('>I', data[0xa0:0xa4])[0] if len(data) > 0xa3 else 0

            # Current beat within bar (byte 0xa6)
            bar_beat = data[0xa6] if len(data) > 0xa6 else 1

            # Create status object
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

            # Store status
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

            # Fire status update callback
            if self.on_status_update:
                self.on_status_update(status)

        except Exception as e:
            logger.error(f"Error parsing CDJ status: {e}")

    def _parse_beat_packet(self, data: bytes):
        """Parse a beat packet."""
        try:
            if len(data) < 40:
                return

            device_number = data[0x21]
            beat = data[0x28] if len(data) > 0x28 else 1

            if self.on_beat:
                self.on_beat(device_number, beat)

        except Exception as e:
            logger.debug(f"Error parsing beat packet: {e}")

    def get_deck_status(self, deck: int) -> Optional[DeckStatus]:
        """Get the current status of a deck."""
        return self._deck_status.get(deck)

    def get_all_deck_status(self) -> Dict[int, DeckStatus]:
        """Get status of all decks."""
        return dict(self._deck_status)

    def get_master_deck(self) -> Optional[int]:
        """Get the deck number that is currently master."""
        for deck, status in self._deck_status.items():
            if status.is_master:
                return deck
        return None

    @property
    def is_database_polling_mode(self) -> bool:
        """True if we're in fallback database polling mode."""
        return self._use_database_polling

    def get_playing_decks(self) -> List[int]:
        """Get list of decks that are currently playing."""
        return [deck for deck, status in self._deck_status.items() if status.is_playing]

    def estimate_position_ms(self, deck: int) -> Optional[int]:
        """
        Estimate the current playback position in milliseconds.

        Uses beat count and BPM to estimate position.
        This is approximate - for precise sync use beat count directly.
        """
        status = self._deck_status.get(deck)
        if not status or status.bpm <= 0:
            return None

        # Calculate ms per beat
        ms_per_beat = 60000.0 / status.bpm

        # Adjust for time since last status packet
        elapsed = time.time() - status.packet_time

        # Estimate position
        if status.is_playing:
            adjusted_beats = status.beat_count + (elapsed * status.bpm / 60.0)
        else:
            adjusted_beats = status.beat_count

        return int(adjusted_beats * ms_per_beat)


# Convenience function for testing
def main():
    """Test the Pro DJ Link listener."""
    logging.basicConfig(level=logging.DEBUG)

    listener = ProLinkListener()

    def on_track(event: TrackLoaded):
        print(f"Track loaded: deck={event.device_number}, id={event.track_id}")

    def on_status(status: DeckStatus):
        if status.is_playing:
            print(f"Deck {status.device_number}: track={status.track_id}, "
                  f"bpm={status.bpm:.1f}, beat={status.beat_count}")

    listener.on_track_loaded = on_track
    listener.on_status_update = on_status

    try:
        listener.start()
        print("Listening for Pro DJ Link packets... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
