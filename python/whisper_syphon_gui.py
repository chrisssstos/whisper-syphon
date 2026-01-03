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
                             QTextEdit, QFrame, QSlider, QProgressBar)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sounddevice as sd
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

# ScreenCaptureKit for system audio (macOS 12.3+)
try:
    from screencapture_audio import ScreenCaptureAudio, check_screencapturekit_available
    HAS_SCREENCAPTURE = check_screencapturekit_available()
except ImportError:
    HAS_SCREENCAPTURE = False
    print("Warning: ScreenCaptureKit module not available")

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
            # Skip completely empty strings only
            if not text:
                return
            # If we had faded, start fresh
            if self.faded:
                self.current_text = ""
                self.faded = False
            # Preserve original text with its spacing (spaces are word boundaries)
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
    level_signal = pyqtSignal(float)  # audio level 0-1
    step_signal = pyqtSignal(int)  # step count for UI


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
        self.screencapture_audio = None  # ScreenCaptureKit capture
        self.use_screencapture = False   # Flag for capture mode
        self.model = None  # Store model for recreating generator
        self.gen = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.other_codebooks = None
        self.renderer = None
        self.ndi = None
        self.block_queue = queue.Queue()
        self.sensitivity = 0.01  # Audio threshold (0-1)
        self.current_level = 0.0  # Current audio level

        # Signals for thread-safe UI updates
        self.signals = SignalEmitter()
        self.signals.text_signal.connect(self.on_text)
        self.signals.status_signal.connect(self.on_status)
        self.signals.level_signal.connect(self.on_level)
        self.signals.step_signal.connect(self.on_step)

        # Timer to clear display after inactivity
        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(self.check_clear_display)
        self.clear_timer.start(100)  # Check every 100ms

        # Watchdog for transcription thread
        self.last_transcribe_time = 0
        self.transcribe_thread = None
        self.needs_reset = False  # Flag for transcribe loop to handle reset
        self.watchdog_timer = QTimer()
        self.watchdog_timer.timeout.connect(self.check_transcribe_health)
        self.watchdog_timer.start(2000)  # Check every 2 seconds

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

        self.step_status = QLabel("Steps: 0")
        self.step_status.setStyleSheet("color: gray;")
        status_layout.addWidget(self.step_status)

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

        # Audio level meter
        level_layout = QHBoxLayout()
        level_label = QLabel("Input Level:")
        level_label.setFont(QFont("Helvetica", 11))
        level_label.setMinimumWidth(80)
        level_layout.addWidget(level_label)

        self.level_bar = QProgressBar()
        self.level_bar.setMinimum(0)
        self.level_bar.setMaximum(100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setMaximumHeight(20)
        self.level_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff00, stop:0.7 #ffff00, stop:1 #ff0000);
                border-radius: 3px;
            }
        """)
        level_layout.addWidget(self.level_bar, 1)

        self.level_value = QLabel("0.00")
        self.level_value.setFont(QFont("Helvetica", 10))
        self.level_value.setMinimumWidth(40)
        level_layout.addWidget(self.level_value)
        layout.addLayout(level_layout)

        # Sensitivity slider
        sens_layout = QHBoxLayout()
        sens_label = QLabel("Threshold:")
        sens_label.setFont(QFont("Helvetica", 11))
        sens_label.setMinimumWidth(80)
        sens_layout.addWidget(sens_label)

        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setMinimum(0)
        self.sens_slider.setMaximum(100)
        self.sens_slider.setValue(1)  # 0.01 default
        self.sens_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #2d2d2d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d4;
                border-radius: 4px;
            }
        """)
        self.sens_slider.valueChanged.connect(self.on_sensitivity_changed)
        sens_layout.addWidget(self.sens_slider, 1)

        self.sens_value = QLabel("0.01")
        self.sens_value.setFont(QFont("Helvetica", 10))
        self.sens_value.setMinimumWidth(40)
        sens_layout.addWidget(self.sens_value)
        layout.addLayout(sens_layout)

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
        # Add ScreenCaptureKit option first if available
        screencapture_idx = -1
        if HAS_SCREENCAPTURE:
            self.device_combo.addItem("🖥️ System Audio (ScreenCaptureKit)", "screencapture")
            screencapture_idx = 0

        devices = sd.query_devices()
        blackhole_idx = -1
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = f"[{i}] {dev['name']}"
                if "blackhole" in dev['name'].lower():
                    name += " (Virtual Device)"
                    if blackhole_idx < 0:
                        blackhole_idx = self.device_combo.count()
                self.device_combo.addItem(name, i)

        # Prefer ScreenCaptureKit, then BlackHole
        if screencapture_idx >= 0:
            self.device_combo.setCurrentIndex(screencapture_idx)
        elif blackhole_idx >= 0:
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
                # Store for recreating tokenizer on reset
                self.mimi_weights = mimi_weights
                self.mimi_codebooks = mimi_codebooks

                model.warmup()

                # Store model reference for recreating generator
                self.model = model

                # More sensitive settings for music/lyrics
                self.gen = models.LmGen(
                    model=model,
                    max_steps=8192,
                    text_sampler=utils.Sampler(top_k=50, temp=0.3),  # Higher temp for lyrics
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
        device_data = self.device_combo.currentData()

        if self.ndi and self.ndi.start():
            self.ndi_status.setText("NDI: On")
            self.ndi_status.setStyleSheet("color: #00ff00;")

        self.running = True
        threading.Thread(target=self.render_loop, daemon=True).start()

        # Check if using ScreenCaptureKit or sounddevice
        if device_data == "screencapture":
            self.use_screencapture = True
            self.screencapture_audio = ScreenCaptureAudio(sample_rate=24000, channels=1)

            # Buffer to accumulate samples to match expected block size
            self._sc_buffer = np.array([], dtype=np.float32)
            self._sc_block_size = 1920  # Match sounddevice block size

            def screencapture_callback(audio_data):
                # Calculate and emit audio level
                level = np.abs(audio_data).mean()
                self.signals.level_signal.emit(level)

                # Accumulate audio data and emit in consistent block sizes
                self._sc_buffer = np.concatenate([self._sc_buffer, audio_data])
                while len(self._sc_buffer) >= self._sc_block_size:
                    block = self._sc_buffer[:self._sc_block_size]
                    self._sc_buffer = self._sc_buffer[self._sc_block_size:]
                    # Always queue audio - model handles silence
                    self.block_queue.put(block.reshape(-1, 1))

            try:
                self.screencapture_audio.start(callback=screencapture_callback)
                self.audio_status.setText("Audio: System (SCK)")
                self.audio_status.setStyleSheet("color: #00ff00;")
            except Exception as e:
                self.transcript.append(f"ScreenCaptureKit error: {e}")
                self.audio_status.setText("Audio: Error")
                self.audio_status.setStyleSheet("color: red;")
                self.running = False
                return
        else:
            self.use_screencapture = False
            device_idx = device_data

            def audio_callback(indata, _frames, _time, _status):
                # Calculate and emit audio level
                level = np.abs(indata).mean()
                self.signals.level_signal.emit(level)

                # Always queue audio - model handles silence
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

        self.last_transcribe_time = time.time()  # Initialize watchdog
        self.transcribe_thread = threading.Thread(target=self.transcribe_loop, daemon=True)
        self.transcribe_thread.start()

        self.start_btn.setText("Stop")
        self.start_btn.setStyleSheet("background-color: #d41a1a;")

    def stop_capture(self):
        self.running = False

        # Stop sounddevice stream
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

        # Stop ScreenCaptureKit
        if self.screencapture_audio:
            self.screencapture_audio.stop()
            self.screencapture_audio = None

        if self.ndi:
            self.ndi.stop()

        self.audio_status.setText("Audio: Off")
        self.audio_status.setStyleSheet("color: gray;")
        self.ndi_status.setText("NDI: Off")
        self.ndi_status.setStyleSheet("color: gray;")
        self.start_btn.setText("Start")
        self.start_btn.setStyleSheet("background-color: #0078d4;")

        # Reset level meter and step counter
        self.level_bar.setValue(0)
        self.level_value.setText("0.00")
        self.step_status.setText("Steps: 0")
        self.step_status.setStyleSheet("color: gray;")

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
        import sys
        step_count = 0
        max_steps = 7500  # Reset well before hitting 8192 limit
        blocks_processed = 0
        last_debug_time = time.time()

        while self.running:
            try:
                # Check if watchdog requested a reset
                if self.needs_reset:
                    print("[TRANSCRIBE] Watchdog requested reset", flush=True)
                    self._reset_generator()
                    step_count = 0
                    self.needs_reset = False
                    continue

                block = self.block_queue.get(timeout=0.1)
                block = block[None, :, 0]

                other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, :self.other_codebooks
                ]

                text_token = self.gen.step(other_audio_tokens[0])
                text_token = text_token[0].item()
                step_count += 1
                blocks_processed += 1
                self.last_transcribe_time = time.time()  # Heartbeat for watchdog

                # Debug output every 5 seconds and update UI
                if time.time() - last_debug_time > 5.0:
                    audio_level = np.abs(block).mean()
                    print(f"[DEBUG] Blocks: {blocks_processed}, Steps: {step_count}, Q: {self.block_queue.qsize()}, Level: {audio_level:.4f}, Shape: {block.shape}, Token: {text_token}", flush=True)
                    self.signals.step_signal.emit(step_count)
                    last_debug_time = time.time()

                # Token 0 = silence, Token 3 = end/pad
                if text_token not in (0, 3):
                    text = self.text_tokenizer.id_to_piece(text_token)
                    text = text.replace("▁", " ")
                    print(f"[TEXT] '{text}'", flush=True)
                    self.signals.text_signal.emit(text)

                # Recreate generator before hitting limit
                if step_count >= max_steps:
                    print(f"[DEBUG] Reached {step_count} steps, resetting...")
                    self._reset_generator()
                    step_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                # On error, try to reset and continue
                print(f"[Transcribe] Error: {e}, resetting...")
                import traceback
                traceback.print_exc()
                self._reset_generator()
                step_count = 0

    def _reset_generator(self):
        """Reset the generator and audio tokenizer, clearing stale audio"""
        print("[RESET] Starting reset...", flush=True)

        # Clear the queue to prevent stale audio
        cleared = 0
        while not self.block_queue.empty():
            try:
                self.block_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        print(f"[RESET] Cleared {cleared} blocks from queue", flush=True)

        # Recreate generator with fresh state (same sensitive settings)
        print("[RESET] Creating new LmGen...", flush=True)
        self.gen = models.LmGen(
            model=self.model,
            max_steps=8192,
            text_sampler=utils.Sampler(top_k=50, temp=0.3),  # Higher temp for lyrics
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )

        # Recreate audio tokenizer (reset() doesn't clear internal step counter)
        print(f"[RESET] Creating new Tokenizer from {self.mimi_weights}...", flush=True)
        self.audio_tokenizer = rustymimi.Tokenizer(self.mimi_weights, num_codebooks=self.mimi_codebooks)
        print("[RESET] Complete!", flush=True)

    def on_text(self, text):
        if self.renderer:
            self.renderer.add_text(text)
            current, visible = self.renderer.get_display_text()
            if visible:
                self.current_word.setText(current)

        self.transcript.insertPlainText(text)
        self.transcript.ensureCursorVisible()

        # Trim transcript if it gets too long (keep last 5000 characters)
        content = self.transcript.toPlainText()
        if len(content) > 6000:
            self.transcript.setPlainText(content[-5000:])

    def on_level(self, level):
        """Update audio level meter"""
        self.current_level = level
        # Scale for display (log scale for better visualization)
        display_level = min(100, int(level * 500))  # Amplify for visibility
        self.level_bar.setValue(display_level)
        self.level_value.setText(f"{level:.2f}")

        # Color the level indicator based on threshold
        if level < self.sensitivity:
            self.level_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #2d2d2d;
                }
                QProgressBar::chunk {
                    background-color: #666;
                    border-radius: 3px;
                }
            """)
        else:
            self.level_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #2d2d2d;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00ff00, stop:0.7 #ffff00, stop:1 #ff0000);
                    border-radius: 3px;
                }
            """)

    def on_sensitivity_changed(self, value):
        """Update sensitivity threshold"""
        self.sensitivity = value / 100.0
        self.sens_value.setText(f"{self.sensitivity:.2f}")

    def on_step(self, step_count):
        """Update step counter in UI"""
        self.step_status.setText(f"Steps: {step_count}")
        self.step_status.setStyleSheet("color: #00ff00;")

    def check_clear_display(self):
        if self.renderer:
            current, visible = self.renderer.get_display_text()
            if not visible:
                self.current_word.setText("")

    def check_transcribe_health(self):
        """Watchdog to restart transcriber if it's stuck"""
        if not self.running:
            return

        # Check if transcriber has been active recently
        now = time.time()
        if self.last_transcribe_time > 0 and now - self.last_transcribe_time > 10:
            print(f"[WATCHDOG] Transcriber stalled for {now - self.last_transcribe_time:.1f}s, signaling reset...", flush=True)
            self.step_status.setText("Steps: STALLED")
            self.step_status.setStyleSheet("color: red;")

            # Signal transcribe loop to reset (don't call _reset_generator directly - threading issue)
            self.needs_reset = True
            self.last_transcribe_time = now

            # Restart transcribe thread if it died
            if self.transcribe_thread and not self.transcribe_thread.is_alive():
                print("[WATCHDOG] Transcribe thread died, restarting...", flush=True)
                self.transcribe_thread = threading.Thread(target=self.transcribe_loop, daemon=True)
                self.transcribe_thread.start()

    def closeEvent(self, event):
        self.running = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        if self.screencapture_audio:
            self.screencapture_audio.stop()
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
