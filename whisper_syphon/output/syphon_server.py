"""Syphon output server."""

import numpy as np
from typing import Optional
import threading
import time


class SyphonOutput:
    """Publishes frames to Syphon for Resolume."""

    def __init__(self, server_name: str = "Whisper Lyrics",
                 width: int = 1920, height: int = 1080, fps: int = 30):
        """Initialize Syphon output.

        Args:
            server_name: Name visible in Resolume.
            width: Frame width.
            height: Frame height.
            fps: Target frame rate.
        """
        self.server_name = server_name
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps

        self._server = None
        self._texture = None
        self._running = False
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Import syphon here to allow graceful failure
        self._syphon_available = False
        try:
            import syphon
            self._syphon = syphon
            self._syphon_available = True
        except Exception as e:
            print(f"Warning: syphon-python not available ({e}), Syphon output disabled")

    def start(self) -> bool:
        """Start Syphon server.

        Returns:
            True if started successfully.
        """
        if not self._syphon_available:
            print("Syphon not available - running in preview-only mode")
            return False

        if self._running:
            return True

        try:
            # Create Syphon Metal server
            self._server = self._syphon.SyphonMetalServer(self.server_name)

            # Create Metal texture
            import Metal
            descriptor = Metal.MTLTextureDescriptor.\
                texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                    Metal.MTLPixelFormatRGBA8Unorm,
                    self.width, self.height, False
                )
            descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)

            self._texture = self._server.device.newTextureWithDescriptor_(descriptor)
            self._region = Metal.MTLRegion((0, 0, 0), (self.width, self.height, 1))
            self._bytes_per_row = self.width * 4

            self._running = True
            self._thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._thread.start()

            print(f"Syphon server '{self.server_name}' started")
            return True

        except Exception as e:
            print(f"Failed to start Syphon: {e}")
            return False

    def stop(self) -> None:
        """Stop Syphon server."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._server:
            try:
                self._server.stop()
            except Exception:
                pass
            self._server = None

        print("Syphon server stopped")

    def set_frame(self, frame: np.ndarray) -> None:
        """Set the current frame to publish.

        Args:
            frame: RGBA numpy array of shape (height, width, 4).
        """
        with self._frame_lock:
            self._current_frame = frame

    def _publish_loop(self) -> None:
        """Publishing loop running at target fps."""
        last_time = time.time()

        while self._running:
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= self.frame_interval:
                last_time = current_time
                self._publish_frame()
            else:
                time.sleep(self.frame_interval - elapsed)

    def _publish_frame(self) -> None:
        """Publish current frame to Syphon."""
        with self._frame_lock:
            frame = self._current_frame

        if frame is None or self._texture is None:
            return

        try:
            # Ensure correct shape and type
            if frame.shape != (self.height, self.width, 4):
                return
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Copy to Metal texture
            self._texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                self._region, 0,
                frame.tobytes(),
                self._bytes_per_row
            )

            # Publish
            self._server.publish_frame_texture(self._texture)

        except Exception as e:
            print(f"Syphon publish error: {e}")

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
