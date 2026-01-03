#!/usr/bin/env python3
"""
Whisper Syphon GUI - Real-time lyrics/speech to NDI using Moshi STT
"""

import sys
import json
import queue
import threading
import time
import numpy as np
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QTextEdit, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sounddevice as sd
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

# NDI output
try:
    from cyndilib.sender import Sender
    from cyndilib.video_frame import VideoSendFrame
    from cyndilib.wrapper.ndi_structs import FourCC
    HAS_NDI = True
except ImportError:
    HAS_NDI = False
    print("Warning: cyndilib not available, NDI output disabled")


class TextRenderer:
    """Renders text to RGBA frames for NDI output"""

    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.current_text = ""
        self.lock = threading.Lock()
        self.last_update = time.time()
        self.fade_timeout = 1.0  # seconds before text fades
        self.max_words = 3  # only show last N words
        self.faded = False  # track if we've already faded

        self.font_large = None
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in font_paths:
            try:
                self.font_large = ImageFont.truetype(path, 120)
                break
            except:
                continue

        if not self.font_large:
            self.font_large = ImageFont.load_default()

    def add_text(self, text: str):
        with self.lock:
            text = text.strip()
            if not text:
                return
            # If we had faded, start fresh
            if self.faded:
                self.current_text = ""
                self.faded = False
            self.current_text += text
            # Keep only last N words
            words = self.current_text.split()
            if len(words) > self.max_words:
                self.current_text = " ".join(words[-self.max_words:])
            self.last_update = time.time()

    def clear(self):
        with self.lock:
            self.current_text = ""

    def get_display_text(self):
        with self.lock:
            # Check if text should fade out
            if time.time() - self.last_update > self.fade_timeout:
                return "", False
            return self.current_text, True

    def render(self) -> np.ndarray:
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        with self.lock:
            elapsed = time.time() - self.last_update
            if elapsed > self.fade_timeout:
                # Text has faded out - mark as faded and clear
                self.faded = True
                self.current_text = ""
                return np.array(img)

            current = self.current_text.strip()

            # Calculate fade alpha (fade out over 0.3 seconds before timeout)
            fade_start = self.fade_timeout - 0.3
            if elapsed > fade_start:
                fade_progress = (self.fade_timeout - elapsed) / 0.3
                alpha = int(255 * max(0, min(1, fade_progress)))
            else:
                alpha = 255

        if current and alpha > 0:
            bbox = draw.textbbox((0, 0), current, font=self.font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2

            # Draw outline
            outline_alpha = int(alpha * 0.8)
            for dx in [-3, 0, 3]:
                for dy in [-3, 0, 3]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), current, font=self.font_large,
                                 fill=(0, 0, 0, outline_alpha))
            # Draw text
            draw.text((x, y), current, font=self.font_large, fill=(255, 255, 255, alpha))

        return np.array(img)


class NDIOutput:
    """NDI video output sender"""

    def __init__(self, name="Whisper Lyrics", width=1920, height=1080, fps=30):
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
    text_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str)  # key, value


class WhisperSyphonGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Syphon (Moshi + NDI)")
        self.setMinimumSize(700, 550)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: white; }
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
            QPushButton:disabled { background-color: #555; }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #444;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #0078d4;
            }
        """)

        # State
        self.running = False
        self.model_loaded = False
        self.audio_stream = None
        self.gen = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.other_codebooks = None
        self.renderer = None
        self.ndi = None
        self.block_queue = queue.Queue()

        # Signals for thread-safe UI updates
        self.signals = SignalEmitter()
        self.signals.text_signal.connect(self.on_text)
        self.signals.status_signal.connect(self.on_status)

        # Timer to clear display after inactivity
        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(self.check_clear_display)
        self.clear_timer.start(100)  # Check every 100ms

        self.setup_ui()
        self.load_devices()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Whisper Syphon")
        title.setFont(QFont("Helvetica", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Audio device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Audio Source:")
        device_label.setFont(QFont("Helvetica", 12))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(400)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo, 1)
        layout.addLayout(device_layout)

        # Status indicators
        status_layout = QHBoxLayout()

        self.model_status = QLabel("Model: Not Loaded")
        self.model_status.setStyleSheet("color: orange;")
        status_layout.addWidget(self.model_status)

        self.ndi_status = QLabel("NDI: Off")
        self.ndi_status.setStyleSheet("color: gray;")
        status_layout.addWidget(self.ndi_status)

        self.audio_status = QLabel("Audio: Off")
        self.audio_status.setStyleSheet("color: gray;")
        status_layout.addWidget(self.audio_status)

        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        button_layout.addWidget(self.load_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_capture)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Current word display
        self.current_word = QLabel("")
        self.current_word.setFont(QFont("Helvetica", 48, QFont.Weight.Bold))
        self.current_word.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_word.setMinimumHeight(100)
        self.current_word.setStyleSheet("background-color: #2d2d2d; border-radius: 8px; padding: 20px;")
        layout.addWidget(self.current_word)

        # Transcript
        transcript_label = QLabel("Transcript:")
        transcript_label.setFont(QFont("Helvetica", 12))
        layout.addWidget(transcript_label)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setFont(QFont("Helvetica", 11))
        self.transcript.setMinimumHeight(120)
        layout.addWidget(self.transcript)

        # NDI info
        ndi_info = QLabel("NDI Output: 'Whisper Lyrics' (look for it in Resolume)")
        ndi_info.setStyleSheet("color: #888;")
        ndi_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ndi_info)

    def load_devices(self):
        devices = sd.query_devices()
        blackhole_idx = -1
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = f"[{i}] {dev['name']}"
                if "blackhole" in dev['name'].lower():
                    name += " (System Audio)"
                    if blackhole_idx < 0:
                        blackhole_idx = self.device_combo.count()
                self.device_combo.addItem(name, i)

        if blackhole_idx >= 0:
            self.device_combo.setCurrentIndex(blackhole_idx)

    def load_model(self):
        self.load_btn.setEnabled(False)
        self.model_status.setText("Model: Loading...")
        self.model_status.setStyleSheet("color: yellow;")

        def load_thread():
            try:
                hf_repo = "kyutai/stt-1b-en_fr-mlx"

                lm_config_path = hf_hub_download(hf_repo, "config.json")
                with open(lm_config_path, "r") as f:
                    lm_config_dict = json.load(f)

                mimi_weights = hf_hub_download(hf_repo, lm_config_dict["mimi_name"])
                moshi_name = lm_config_dict.get("moshi_name", "model.safetensors")
                moshi_weights = hf_hub_download(hf_repo, moshi_name)
                tokenizer_path = hf_hub_download(hf_repo, lm_config_dict["tokenizer_name"])

                lm_config = models.LmConfig.from_config_dict(lm_config_dict)
                model = models.Lm(lm_config)
                model.set_dtype(mx.bfloat16)

                if moshi_weights.endswith(".q4.safetensors"):
                    nn.quantize(model, bits=4, group_size=32)
                elif moshi_weights.endswith(".q8.safetensors"):
                    nn.quantize(model, bits=8, group_size=64)

                model.load_weights(moshi_weights, strict=True)

                self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

                generated_codebooks = lm_config.generated_codebooks
                self.other_codebooks = lm_config.other_codebooks
                mimi_codebooks = max(generated_codebooks, self.other_codebooks)
                self.audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)

                model.warmup()

                self.gen = models.LmGen(
                    model=model,
                    max_steps=8192,
                    text_sampler=utils.Sampler(top_k=25, temp=0),
                    audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                    check=False,
                )

                self.renderer = TextRenderer(1920, 1080)
                self.ndi = NDIOutput("Whisper Lyrics", 1920, 1080, 30)

                self.model_loaded = True
                self.signals.status_signal.emit("model", "ready")

            except Exception as e:
                self.signals.status_signal.emit("model", f"error:{e}")

        threading.Thread(target=load_thread, daemon=True).start()

    def on_status(self, key, value):
        if key == "model":
            if value == "ready":
                self.model_status.setText("Model: Ready")
                self.model_status.setStyleSheet("color: #00ff00;")
                self.start_btn.setEnabled(True)
                self.load_btn.setEnabled(False)
            elif value.startswith("error:"):
                self.model_status.setText("Model: Error")
                self.model_status.setStyleSheet("color: red;")
                self.load_btn.setEnabled(True)
                self.transcript.append(f"Error: {value[6:]}")

    def toggle_capture(self):
        if not self.running:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        device_idx = self.device_combo.currentData()

        if self.ndi and self.ndi.start():
            self.ndi_status.setText("NDI: On")
            self.ndi_status.setStyleSheet("color: #00ff00;")

        self.running = True
        threading.Thread(target=self.render_loop, daemon=True).start()

        def audio_callback(indata, _frames, _time, _status):
            self.block_queue.put(indata.copy())

        self.audio_stream = sd.InputStream(
            device=device_idx,
            channels=1,
            dtype="float32",
            samplerate=24000,
            blocksize=1920,
            callback=audio_callback,
        )
        self.audio_stream.start()
        self.audio_status.setText("Audio: On")
        self.audio_status.setStyleSheet("color: #00ff00;")

        threading.Thread(target=self.transcribe_loop, daemon=True).start()

        self.start_btn.setText("Stop")
        self.start_btn.setStyleSheet("background-color: #d41a1a;")

    def stop_capture(self):
        self.running = False

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

        if self.ndi:
            self.ndi.stop()

        self.audio_status.setText("Audio: Off")
        self.audio_status.setStyleSheet("color: gray;")
        self.ndi_status.setText("NDI: Off")
        self.ndi_status.setStyleSheet("color: gray;")
        self.start_btn.setText("Start")
        self.start_btn.setStyleSheet("background-color: #0078d4;")

    def render_loop(self):
        frame_time = 1.0 / 30
        while self.running:
            start = time.time()
            if self.renderer and self.ndi:
                frame = self.renderer.render()
                self.ndi.send_frame(frame)
            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    def transcribe_loop(self):
        while self.running:
            try:
                block = self.block_queue.get(timeout=0.1)
                block = block[None, :, 0]

                other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, :self.other_codebooks
                ]

                text_token = self.gen.step(other_audio_tokens[0])
                text_token = text_token[0].item()

                if text_token not in (0, 3):
                    text = self.text_tokenizer.id_to_piece(text_token)
                    text = text.replace("▁", " ")
                    self.signals.text_signal.emit(text)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")

    def on_text(self, text):
        if self.renderer:
            self.renderer.add_text(text)
            current, visible = self.renderer.get_display_text()
            if visible:
                self.current_word.setText(current)

        self.transcript.insertPlainText(text)
        self.transcript.ensureCursorVisible()

    def check_clear_display(self):
        if self.renderer:
            current, visible = self.renderer.get_display_text()
            if not visible:
                self.current_word.setText("")

    def closeEvent(self, event):
        self.running = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        if self.ndi:
            self.ndi.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = WhisperSyphonGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
