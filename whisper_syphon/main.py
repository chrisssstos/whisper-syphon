"""Main application entry point."""

import sys
from queue import Queue, Empty
import threading
import time

from .audio.devices import get_input_devices
from .audio.capture import AudioCapture
from .transcription.processor import StreamingProcessor
from .display.renderer import TextRenderer
from .display.window import MainWindow

# Lazy import for SyphonOutput to avoid crash on macOS version check
SyphonOutput = None
def _get_syphon_output():
    global SyphonOutput
    if SyphonOutput is None:
        try:
            from .output.syphon_server import SyphonOutput as _SyphonOutput
            SyphonOutput = _SyphonOutput
        except Exception as e:
            print(f"Syphon import failed: {e}")
            SyphonOutput = None
    return SyphonOutput


class WhisperSyphonApp:
    """Main application coordinating all components."""

    WIDTH = 1920
    HEIGHT = 1080
    FPS = 30

    def __init__(self):
        """Initialize application."""
        self.audio_queue = Queue()
        self.word_queue = Queue()

        self.capture: AudioCapture = None
        self.processor: StreamingProcessor = None
        self.renderer = TextRenderer(self.WIDTH, self.HEIGHT)

        # Try to create Syphon output (may fail on some macOS versions)
        SyphonClass = _get_syphon_output()
        if SyphonClass:
            self.syphon = SyphonClass("Whisper Lyrics", self.WIDTH, self.HEIGHT, self.FPS)
        else:
            self.syphon = None

        self.window: MainWindow = None

        self._running = False
        self._output_thread: threading.Thread = None
        self._current_word = ""

    def start(self, device_index: int) -> None:
        """Start processing.

        Args:
            device_index: Audio input device index.
        """
        if self._running:
            return

        self._running = True

        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        while not self.word_queue.empty():
            try:
                self.word_queue.get_nowait()
            except Empty:
                break

        # Start audio capture
        self.capture = AudioCapture(device_index, self.audio_queue)
        self.capture.start()

        # Start transcription processor
        self.processor = StreamingProcessor(
            self.audio_queue, self.word_queue, model_size="base"
        )
        self.processor.start()

        # Start Syphon output
        if self.syphon:
            syphon_started = self.syphon.start()
            if syphon_started:
                self.window.set_status("Syphon: Active | Listening...")
            else:
                self.window.set_status("Syphon: Unavailable | Listening...")
        else:
            self.window.set_status("Syphon: Not loaded | Listening...")

        # Start output thread
        self._output_thread = threading.Thread(target=self._output_loop, daemon=True)
        self._output_thread.start()

        # Start word polling
        self._poll_words()

    def stop(self) -> None:
        """Stop processing."""
        self._running = False

        if self.capture:
            self.capture.stop()
            self.capture = None

        if self.processor:
            self.processor.stop()
            self.processor = None

        if self.syphon:
            self.syphon.stop()

        if self._output_thread:
            self._output_thread.join(timeout=1.0)
            self._output_thread = None

        self._current_word = ""

    def _poll_words(self) -> None:
        """Poll word queue and update display."""
        if not self._running:
            return

        try:
            while True:
                word_info = self.word_queue.get_nowait()
                word = word_info['text']
                if word:
                    self._current_word = word
                    self.window.update_word(word)
        except Empty:
            pass

        # Schedule next poll
        self.window.schedule(self._poll_words, 50)

    def _output_loop(self) -> None:
        """Output loop for rendering and Syphon publishing."""
        frame_interval = 1.0 / self.FPS
        last_word = ""

        while self._running:
            start_time = time.time()

            # Render current word
            word = self._current_word
            if word != last_word:
                frame = self.renderer.render(word)
                last_word = word
            else:
                frame = self.renderer.render(word)

            # Update Syphon
            if self.syphon:
                self.syphon.set_frame(frame)

            # Update preview (throttled)
            if hasattr(self, '_preview_counter'):
                self._preview_counter += 1
            else:
                self._preview_counter = 0

            if self._preview_counter % 3 == 0:  # Every 3rd frame
                self.window.schedule(lambda f=frame: self.window.update_preview(f))

            # Maintain frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def run(self) -> None:
        """Run the application."""
        # Get audio devices
        devices = get_input_devices()
        if not devices:
            print("Error: No audio input devices found")
            sys.exit(1)

        print(f"Found {len(devices)} audio input device(s)")

        # Create window
        self.window = MainWindow(
            devices,
            on_start=self.start,
            on_stop=self.stop
        )

        # Initialize with black frame
        black_frame = self.renderer.get_black_frame()
        self.window.update_preview(black_frame)

        # Run main loop
        print("Starting Whisper Syphon...")
        self.window.run()


def main():
    """Entry point."""
    app = WhisperSyphonApp()
    app.run()


if __name__ == "__main__":
    main()
