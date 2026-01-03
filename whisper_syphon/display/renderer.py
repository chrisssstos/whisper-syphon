"""Text renderer using PIL."""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Optional, Tuple


class TextRenderer:
    """Renders text to numpy RGBA arrays for Syphon output."""

    def __init__(self, width: int = 1920, height: int = 1080,
                 font_size: int = 200, font_path: Optional[str] = None):
        """Initialize text renderer.

        Args:
            width: Output width in pixels.
            height: Output height in pixels.
            font_size: Font size in points.
            font_path: Path to TTF font file, or None for system font.
        """
        self.width = width
        self.height = height
        self.font_size = font_size

        # Load font
        if font_path:
            self.font = ImageFont.truetype(font_path, font_size)
        else:
            # Try common system fonts
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSDisplay.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
            self.font = None
            for path in font_paths:
                try:
                    self.font = ImageFont.truetype(path, font_size)
                    break
                except (IOError, OSError):
                    continue

            if self.font is None:
                # Fall back to default
                self.font = ImageFont.load_default()

        # Pre-create black background
        self._background = Image.new('RGBA', (width, height), (0, 0, 0, 255))
        self._black_frame = np.array(self._background)

    def render(self, text: str, color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> np.ndarray:
        """Render text centered on black background.

        Args:
            text: Text to render.
            color: RGBA color tuple.

        Returns:
            Numpy array of shape (height, width, 4) dtype uint8.
        """
        if not text:
            return self._black_frame.copy()

        # Create fresh image
        img = self._background.copy()
        draw = ImageDraw.Draw(img)

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center position
        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2

        # Draw text
        draw.text((x, y), text, font=self.font, fill=color)

        return np.array(img)

    def get_black_frame(self) -> np.ndarray:
        """Get a black frame (no text)."""
        return self._black_frame.copy()
