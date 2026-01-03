"""
ScreenCaptureKit Audio Capture for macOS 12.3+
Captures system audio natively without third-party virtual audio devices.
"""

import threading
import queue
import numpy as np
from typing import Optional, Callable

import objc
from Foundation import NSObject, NSRunLoop, NSDate, NSDefaultRunLoopMode

# Load ScreenCaptureKit framework
SCK = objc.loadBundle(
    'ScreenCaptureKit',
    globals(),
    bundle_path='/System/Library/Frameworks/ScreenCaptureKit.framework'
)

# Import classes
SCShareableContent = objc.lookUpClass('SCShareableContent')
SCContentFilter = objc.lookUpClass('SCContentFilter')
SCStreamConfiguration = objc.lookUpClass('SCStreamConfiguration')
SCStream = objc.lookUpClass('SCStream')

# Register selectors with block signatures
objc.registerMetaDataForSelector(
    b'SCShareableContent',
    b'getShareableContentWithCompletionHandler:',
    {
        'arguments': {
            2: {
                'callable': {
                    'retval': {'type': b'v'},
                    'arguments': {
                        0: {'type': b'^v'},
                        1: {'type': b'@'},
                        2: {'type': b'@'},
                    }
                }
            }
        }
    }
)

objc.registerMetaDataForSelector(
    b'SCStream',
    b'startCaptureWithCompletionHandler:',
    {
        'arguments': {
            2: {
                'callable': {
                    'retval': {'type': b'v'},
                    'arguments': {
                        0: {'type': b'^v'},
                        1: {'type': b'@'},
                    }
                }
            }
        }
    }
)

objc.registerMetaDataForSelector(
    b'SCStream',
    b'stopCaptureWithCompletionHandler:',
    {
        'arguments': {
            2: {
                'callable': {
                    'retval': {'type': b'v'},
                    'arguments': {
                        0: {'type': b'^v'},
                        1: {'type': b'@'},
                    }
                }
            }
        }
    }
)


class StreamOutputDelegate(NSObject):
    """Delegate to receive audio samples from SCStream"""

    def init(self):
        self = objc.super(StreamOutputDelegate, self).init()
        if self is None:
            return None
        self._callback = None
        return self

    @objc.python_method
    def setCallback_(self, callback):
        self._callback = callback

    # Protocol method for SCStreamOutput
    @objc.typedSelector(b'v@:@@q')
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
        """Called when new samples are available. outputType: 0=screen, 1=audio"""
        if outputType != 1:  # Only process audio
            return

        try:
            import CoreMedia

            # Get audio buffer list from sample buffer
            blockBuffer = CoreMedia.CMSampleBufferGetDataBuffer(sampleBuffer)
            if blockBuffer is None:
                return

            # Get the data length
            length = CoreMedia.CMBlockBufferGetDataLength(blockBuffer)
            if length == 0:
                return

            # Extract raw bytes
            # CMBlockBufferCopyDataBytes returns (status, data_bytearray)
            result = CoreMedia.CMBlockBufferCopyDataBytes(
                blockBuffer, 0, length, None
            )

            # Handle different return formats
            if isinstance(result, tuple):
                status, data_out = result
                if status != 0:
                    return
            else:
                # If it returns just the status, we need different approach
                return

            # Convert to float32 numpy array
            # ScreenCaptureKit outputs float32 audio
            audio_data = np.frombuffer(bytes(data_out), dtype=np.float32).copy()

            if self._callback and len(audio_data) > 0:
                self._callback(audio_data)

        except Exception as e:
            pass  # Silently ignore errors to avoid spam


class ScreenCaptureAudio:
    """
    Captures system audio using macOS ScreenCaptureKit.
    Requires macOS 12.3 or later.
    """

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = None
        self.delegate = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.running = False
        self._callback: Optional[Callable] = None
        self._runloop_thread = None
        self._buffer = np.array([], dtype=np.float32)

    def _audio_callback(self, audio_data: np.ndarray):
        """Internal callback that processes audio and queues it"""
        # ScreenCaptureKit may output stereo, convert to mono if needed
        if len(audio_data) % 2 == 0 and self.channels == 1:
            # Assume interleaved stereo, convert to mono
            try:
                stereo = audio_data.reshape(-1, 2)
                audio_data = stereo.mean(axis=1).astype(np.float32)
            except:
                pass

        self.audio_queue.put(audio_data)

        if self._callback:
            self._callback(audio_data)

    def start(self, display_id: Optional[int] = None, callback: Optional[Callable] = None):
        """
        Start capturing system audio.

        Args:
            display_id: Optional display ID. If None, uses main display.
            callback: Optional callback(audio_data: np.ndarray) for each audio chunk.
        """
        if self.running:
            return

        self._callback = callback

        # Get shareable content synchronously
        event = threading.Event()
        result = {'content': None, 'error': None}

        def content_handler(content, error):
            if error:
                result['error'] = error
            else:
                result['content'] = content
            event.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(content_handler)

        # Run the runloop briefly to let the completion handler fire
        for _ in range(50):  # Try for up to 5 seconds
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode,
                NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )
            if event.is_set():
                break

        if not event.is_set():
            raise RuntimeError("Timeout getting shareable content")

        if result['error']:
            raise RuntimeError(f"Failed to get shareable content: {result['error']}")

        content = result['content']
        displays = content.displays()
        if not displays or len(displays) == 0:
            raise RuntimeError("No displays available")

        # Select display
        target_display = displays[0]
        if display_id is not None:
            for d in displays:
                if d.displayID() == display_id:
                    target_display = d
                    break

        # Create content filter
        content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            target_display, []
        )

        # Configure stream
        config = SCStreamConfiguration.alloc().init()
        config.setWidth_(2)
        config.setHeight_(2)
        config.setShowsCursor_(False)
        config.setCapturesAudio_(True)
        config.setSampleRate_(self.sample_rate)
        config.setChannelCount_(2)  # Capture stereo, convert to mono later
        config.setExcludesCurrentProcessAudio_(True)

        # Create delegate
        self.delegate = StreamOutputDelegate.alloc().init()
        self.delegate.setCallback_(self._audio_callback)

        # Create stream
        self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, None
        )

        # Add stream output
        error = None
        success = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self.delegate,
            1,  # SCStreamOutputTypeAudio
            None,
            error
        )

        # Start capture
        start_event = threading.Event()
        start_result = {'error': None}

        def start_handler(error):
            if error:
                start_result['error'] = error
            start_event.set()

        self.stream.startCaptureWithCompletionHandler_(start_handler)

        # Wait for start
        for _ in range(50):
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode,
                NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )
            if start_event.is_set():
                break

        if start_result['error']:
            raise RuntimeError(f"Failed to start capture: {start_result['error']}")

        self.running = True

        # Start runloop thread
        self._runloop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._runloop_thread.start()

        print(f"ScreenCaptureKit: capturing system audio at {self.sample_rate}Hz")

    def _run_loop(self):
        """Process callbacks on runloop"""
        while self.running:
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode,
                NSDate.dateWithTimeIntervalSinceNow_(0.01)
            )

    def stop(self):
        """Stop capturing"""
        if not self.running:
            return

        self.running = False

        if self.stream:
            stop_event = threading.Event()

            def stop_handler(error):
                stop_event.set()

            self.stream.stopCaptureWithCompletionHandler_(stop_handler)

            # Wait for stop
            for _ in range(20):
                NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    NSDefaultRunLoopMode,
                    NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )
                if stop_event.is_set():
                    break

            self.stream = None

        self.delegate = None
        print("ScreenCaptureKit: stopped")

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Read audio from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __del__(self):
        self.stop()


def check_screencapturekit_available() -> bool:
    """Check if ScreenCaptureKit is available"""
    try:
        import platform
        version = platform.mac_ver()[0]
        parts = version.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major > 12) or (major == 12 and minor >= 3)
    except:
        return False


if __name__ == "__main__":
    import time

    if not check_screencapturekit_available():
        print("ScreenCaptureKit requires macOS 12.3+")
        exit(1)

    print("Testing ScreenCaptureKit audio capture...")
    capture = ScreenCaptureAudio(sample_rate=24000, channels=1)

    def on_audio(data):
        level = np.abs(data).mean()
        bars = int(level * 50)
        print(f"\rLevel: {'█' * bars}{' ' * (50-bars)} {level:.4f}", end='', flush=True)

    try:
        capture.start(callback=on_audio)
        print("Capturing... Press Ctrl+C to stop")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.stop()
