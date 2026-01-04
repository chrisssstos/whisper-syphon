"""Render synced lyrics with word-level highlighting for NDI output."""

import threading
import time
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lyrics.lrc_parser import LyricLine, LyricWord, LRCParser


class LyricsRenderer:
    """Render synced lyrics with word-level highlighting.

    Shows the current line centered with the active word highlighted,
    plus context lines above and below in smaller font.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        highlight_color: Tuple[int, int, int, int] = (255, 255, 0, 255),
        text_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        sung_color: Tuple[int, int, int, int] = (150, 150, 150, 255),
        upcoming_color: Tuple[int, int, int, int] = (200, 200, 200, 200)
    ):
        """Initialize the lyrics renderer.

        Args:
            width: Output frame width
            height: Output frame height
            highlight_color: RGBA color for current word
            text_color: RGBA color for unsung words in current line
            sung_color: RGBA color for already sung words
            upcoming_color: RGBA color for upcoming lines
        """
        self.width = width
        self.height = height

        # Colors
        self.highlight_color = highlight_color
        self.text_color = text_color
        self.sung_color = sung_color
        self.upcoming_color = upcoming_color
        self.outline_color = (0, 0, 0, 200)

        # State
        self._lyrics: List[LyricLine] = []
        self._position_ms: int = 0
        self._lock = threading.Lock()

        # Display settings
        self.context_lines_before = 1
        self.context_lines_after = 2
        self.line_spacing = 1.5  # Multiplier for line height

        # Load fonts
        self._init_fonts()

    def _init_fonts(self):
        """Initialize fonts for rendering."""
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/Arial.ttf",
        ]

        self.font_main = None
        self.font_context = None

        for path in font_paths:
            try:
                self.font_main = ImageFont.truetype(path, 100)
                self.font_context = ImageFont.truetype(path, 60)
                break
            except Exception:
                continue

        if not self.font_main:
            self.font_main = ImageFont.load_default()
            self.font_context = self.font_main

    def set_lyrics(self, lyrics: List[LyricLine]):
        """Set the lyrics to display.

        Args:
            lyrics: List of LyricLine objects (from LRCParser.parse())
        """
        with self._lock:
            self._lyrics = lyrics
            self._position_ms = 0

    def set_position(self, position_ms: int):
        """Update the current playback position.

        Args:
            position_ms: Current position in milliseconds
        """
        with self._lock:
            self._position_ms = position_ms

    def clear(self):
        """Clear lyrics and reset state."""
        with self._lock:
            self._lyrics = []
            self._position_ms = 0

    def render(self) -> np.ndarray:
        """Render the current frame.

        Returns:
            RGBA numpy array of shape (height, width, 4)
        """
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        with self._lock:
            if not self._lyrics:
                return np.array(img)

            position_ms = self._position_ms
            lyrics = self._lyrics

        # Find current line index
        current_idx = LRCParser.get_current_line_index(lyrics, position_ms)

        if current_idx < 0:
            # Before first line - show upcoming
            current_idx = 0

        # Calculate vertical positions
        center_y = self.height // 2

        # Render current line with word highlighting
        if 0 <= current_idx < len(lyrics):
            current_line = lyrics[current_idx]
            self._render_current_line(draw, current_line, position_ms, center_y)

        # Render context lines above
        y_offset = center_y
        for i in range(1, self.context_lines_before + 1):
            idx = current_idx - i
            if idx >= 0:
                line_height = self._get_line_height(self.font_context)
                y_offset -= int(line_height * self.line_spacing)
                self._render_context_line(draw, lyrics[idx], y_offset, is_past=True)

        # Render context lines below
        y_offset = center_y
        for i in range(1, self.context_lines_after + 1):
            idx = current_idx + i
            if idx < len(lyrics):
                line_height = self._get_line_height(self.font_main if i == 1 else self.font_context)
                y_offset += int(line_height * self.line_spacing)
                self._render_context_line(draw, lyrics[idx], y_offset, is_past=False)

        return np.array(img)

    def _render_current_line(
        self,
        draw: ImageDraw.ImageDraw,
        line: LyricLine,
        position_ms: int,
        y: int
    ):
        """Render the current line with word-level highlighting.

        Args:
            draw: PIL ImageDraw object
            line: Current LyricLine
            position_ms: Current playback position
            y: Vertical center position
        """
        if not line.words:
            # No word timing - render whole line
            self._render_text_centered(
                draw, line.raw_text, y, self.font_main, self.text_color
            )
            return

        # Get current word index
        current_word_idx = LRCParser.get_current_word_index(line, position_ms)

        # Calculate total width for centering
        total_width = 0
        word_widths = []
        space_width = self._get_text_width(" ", self.font_main)

        for word in line.words:
            w = self._get_text_width(word.text, self.font_main)
            word_widths.append(w)
            total_width += w

        # Add space widths
        total_width += space_width * (len(line.words) - 1)

        # Starting x position (centered)
        x = (self.width - total_width) // 2

        # Get text height for vertical centering
        text_height = self._get_line_height(self.font_main)
        y_pos = y - text_height // 2

        # Render each word
        for i, (word, width) in enumerate(zip(line.words, word_widths)):
            # Determine color based on position
            if i < current_word_idx:
                # Already sung
                color = self.sung_color
            elif i == current_word_idx:
                # Current word - highlighted
                color = self.highlight_color
            else:
                # Upcoming
                color = self.text_color

            # Render with outline
            self._render_text_with_outline(
                draw, word.text, x, y_pos, self.font_main, color
            )

            x += width + space_width

    def _render_context_line(
        self,
        draw: ImageDraw.ImageDraw,
        line: LyricLine,
        y: int,
        is_past: bool
    ):
        """Render a context line (above or below current).

        Args:
            draw: PIL ImageDraw object
            line: LyricLine to render
            y: Vertical center position
            is_past: True if this is a past line (above current)
        """
        color = self.sung_color if is_past else self.upcoming_color
        self._render_text_centered(draw, line.raw_text, y, self.font_context, color)

    def _render_text_centered(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        y: int,
        font: ImageFont.FreeTypeFont,
        color: Tuple[int, int, int, int]
    ):
        """Render text centered horizontally.

        Args:
            draw: PIL ImageDraw object
            text: Text to render
            y: Vertical center position
            font: Font to use
            color: RGBA color
        """
        if not text:
            return

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.width - text_width) // 2
        y_pos = y - text_height // 2

        self._render_text_with_outline(draw, text, x, y_pos, font, color)

    def _render_text_with_outline(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        color: Tuple[int, int, int, int],
        outline_width: int = 3
    ):
        """Render text with a dark outline for readability.

        Args:
            draw: PIL ImageDraw object
            text: Text to render
            x: X position
            y: Y position
            font: Font to use
            color: RGBA color for text
            outline_width: Width of outline in pixels
        """
        # Calculate outline alpha based on text alpha
        outline_alpha = int(color[3] * 0.7)
        outline_color = (0, 0, 0, outline_alpha)

        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

        # Draw main text
        draw.text((x, y), text, font=font, fill=color)

    def _get_text_width(self, text: str, font: ImageFont.FreeTypeFont) -> int:
        """Get the width of text in pixels.

        Args:
            text: Text to measure
            font: Font to use

        Returns:
            Width in pixels
        """
        # Create temporary image for measurement
        img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def _get_line_height(self, font: ImageFont.FreeTypeFont) -> int:
        """Get the height of a line of text.

        Args:
            font: Font to use

        Returns:
            Height in pixels
        """
        img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), "Xgjpq", font=font)
        return bbox[3] - bbox[1]

    def get_current_line_text(self) -> str:
        """Get the text of the current line.

        Returns:
            Current line text or empty string
        """
        with self._lock:
            if not self._lyrics:
                return ""

            current_idx = LRCParser.get_current_line_index(self._lyrics, self._position_ms)
            if 0 <= current_idx < len(self._lyrics):
                return self._lyrics[current_idx].raw_text

            return ""
