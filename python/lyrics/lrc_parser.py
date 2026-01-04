"""Parse LRC lyrics format with support for word-level timestamps."""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LyricWord:
    """A single word with timing information."""
    text: str
    start_ms: int
    end_ms: int = 0  # Inferred from next word's start or line end


@dataclass
class LyricLine:
    """A line of lyrics with word-level timing."""
    start_ms: int
    words: List[LyricWord] = field(default_factory=list)
    raw_text: str = ""

    @property
    def end_ms(self) -> int:
        """End time is the last word's end time or start + estimated duration."""
        if self.words and self.words[-1].end_ms > 0:
            return self.words[-1].end_ms
        # Estimate ~50ms per character if no end time
        return self.start_ms + len(self.raw_text) * 50


class LRCParser:
    """Parse LRC format lyrics with support for enhanced word-level timestamps.

    Standard LRC format:
        [00:15.20] This is a line

    Enhanced LRC format (word-level):
        [00:15.20] <00:15.20>This <00:15.50>is <00:15.80>a <00:16.00>line
    """

    # Regex patterns
    LINE_TIMESTAMP = re.compile(r'\[(\d{2}):(\d{2})\.(\d{2,3})\]')
    WORD_TIMESTAMP = re.compile(r'<(\d{2}):(\d{2})\.(\d{2,3})>')

    @staticmethod
    def _parse_timestamp(minutes: str, seconds: str, centis: str) -> int:
        """Convert timestamp components to milliseconds."""
        ms = int(minutes) * 60 * 1000 + int(seconds) * 1000
        # Handle both centiseconds (2 digits) and milliseconds (3 digits)
        if len(centis) == 2:
            ms += int(centis) * 10
        else:
            ms += int(centis)
        return ms

    @classmethod
    def parse(cls, lrc_content: str) -> List[LyricLine]:
        """Parse LRC content into structured LyricLine objects.

        Args:
            lrc_content: Raw LRC file content

        Returns:
            List of LyricLine objects sorted by start time
        """
        if not lrc_content:
            return []

        lines = []

        for raw_line in lrc_content.strip().split('\n'):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            # Skip metadata lines like [ar:Artist]
            if raw_line.startswith('[') and ':' in raw_line.split(']')[0]:
                bracket_content = raw_line.split(']')[0][1:]
                if not bracket_content[0].isdigit():
                    continue

            # Extract line timestamp
            line_match = cls.LINE_TIMESTAMP.match(raw_line)
            if not line_match:
                continue

            line_start_ms = cls._parse_timestamp(
                line_match.group(1),
                line_match.group(2),
                line_match.group(3)
            )

            # Get text after line timestamp
            text_content = raw_line[line_match.end():].strip()

            # Check for enhanced word-level timestamps
            word_matches = list(cls.WORD_TIMESTAMP.finditer(text_content))

            lyric_line = LyricLine(start_ms=line_start_ms, raw_text="")

            if word_matches:
                # Enhanced format with word timestamps
                words = []
                for i, match in enumerate(word_matches):
                    word_start_ms = cls._parse_timestamp(
                        match.group(1),
                        match.group(2),
                        match.group(3)
                    )

                    # Get word text (between this timestamp and next, or end of line)
                    start_pos = match.end()
                    if i + 1 < len(word_matches):
                        end_pos = word_matches[i + 1].start()
                    else:
                        end_pos = len(text_content)

                    word_text = text_content[start_pos:end_pos].strip()
                    if word_text:
                        words.append(LyricWord(
                            text=word_text,
                            start_ms=word_start_ms
                        ))

                # Set end times based on next word's start
                for i in range(len(words) - 1):
                    words[i].end_ms = words[i + 1].start_ms
                if words:
                    # Last word ends at estimated line end
                    words[-1].end_ms = words[-1].start_ms + len(words[-1].text) * 80

                lyric_line.words = words
                lyric_line.raw_text = ' '.join(w.text for w in words)
            else:
                # Standard format - create single word entries per word
                clean_text = text_content
                words_text = clean_text.split()

                if words_text:
                    # Estimate word timing by dividing line duration
                    char_count = sum(len(w) for w in words_text)
                    if char_count == 0:
                        char_count = 1

                    # Assume average speaking rate of ~150 words per minute
                    total_duration = max(len(words_text) * 400, 1000)  # At least 1 second

                    current_ms = line_start_ms
                    words = []
                    for word in words_text:
                        word_duration = int(total_duration * len(word) / char_count)
                        words.append(LyricWord(
                            text=word,
                            start_ms=current_ms,
                            end_ms=current_ms + word_duration
                        ))
                        current_ms += word_duration

                    lyric_line.words = words

                lyric_line.raw_text = clean_text

            lines.append(lyric_line)

        # Sort by start time
        lines.sort(key=lambda x: x.start_ms)
        return lines

    @staticmethod
    def get_current_line(lines: List[LyricLine], position_ms: int) -> Optional[LyricLine]:
        """Get the current line at a given playback position.

        Uses binary search for efficiency.

        Args:
            lines: List of LyricLine objects (must be sorted by start_ms)
            position_ms: Current playback position in milliseconds

        Returns:
            The current LyricLine or None if before first line
        """
        if not lines:
            return None

        # Binary search for the current line
        left, right = 0, len(lines) - 1
        result = None

        while left <= right:
            mid = (left + right) // 2
            if lines[mid].start_ms <= position_ms:
                result = lines[mid]
                left = mid + 1
            else:
                right = mid - 1

        return result

    @staticmethod
    def get_current_line_index(lines: List[LyricLine], position_ms: int) -> int:
        """Get the index of the current line at a given playback position.

        Args:
            lines: List of LyricLine objects (must be sorted by start_ms)
            position_ms: Current playback position in milliseconds

        Returns:
            Index of current line, or -1 if before first line
        """
        if not lines:
            return -1

        left, right = 0, len(lines) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2
            if lines[mid].start_ms <= position_ms:
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result

    @staticmethod
    def get_current_word_index(line: LyricLine, position_ms: int) -> int:
        """Get the index of the currently active word within a line.

        Args:
            line: LyricLine object
            position_ms: Current playback position in milliseconds

        Returns:
            Index of current word within the line, or -1 if before first word
        """
        if not line.words:
            return -1

        for i, word in enumerate(line.words):
            if word.start_ms <= position_ms < word.end_ms:
                return i
            elif word.start_ms > position_ms:
                return max(0, i - 1)

        # Past the last word
        return len(line.words) - 1
