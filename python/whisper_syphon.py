#!/usr/bin/env python3
"""
Whisper Syphon - Real-time lyrics/speech to NDI using Moshi STT
"""

import argparse
import json
import queue
import threading
import time
import numpy as np
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont

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
        self.word_history = []
        self.max_history = 15
        self.lock = threading.Lock()

        # Try to load a nice font
        self.font_large = None
        self.font_small = None
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in font_paths:
            try:
                self.font_large = ImageFont.truetype(path, 100)
                self.font_small = ImageFont.truetype(path, 40)
                break
            except:
                continue

        if not self.font_large:
            self.font_large = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def add_text(self, text: str):
        """Add transcribed text"""
        with self.lock:
            # Clean up the text
            text = text.strip()
            if not text:
                return

            self.current_text += text

            # Split into words and update history
            words = self.current_text.split()
            if len(words) > 3:
                # Move older words to history
                self.word_history.extend(words[:-3])
                self.current_text = " ".join(words[-3:])

                # Trim history
                if len(self.word_history) > self.max_history:
                    self.word_history = self.word_history[-self.max_history:]

    def render(self) -> np.ndarray:
        """Render current state to RGBA numpy array"""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        with self.lock:
            current = self.current_text.strip()
            history = " ".join(self.word_history[-10:])

        if current:
            # Draw current text (large, centered)
            bbox = draw.textbbox((0, 0), current, font=self.font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2

            # White text with black outline
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), current, font=self.font_large,
                                 fill=(0, 0, 0, 255))
            draw.text((x, y), current, font=self.font_large, fill=(255, 255, 255, 255))

        if history:
            # Draw history (smaller, above)
            bbox = draw.textbbox((0, 0), history, font=self.font_small)
            text_width = bbox[2] - bbox[0]
            hx = (self.width - text_width) // 2
            hy = self.height // 2 - 120
            draw.text((hx, hy), history, font=self.font_small, fill=(180, 180, 180, 200))

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
        # Attach frame to sender
        self.sender.set_video_frame(self.video_frame)

    def start(self):
        if self.sender:
            self.sender.open()
            self.running = True
            print(f"NDI sender '{self.name}' started", flush=True)

    def stop(self):
        if self.sender:
            self.sender.close()
            self.running = False

    def send_frame(self, rgba_array: np.ndarray):
        if not self.sender or not self.running:
            return
        if rgba_array.shape != (self.height, self.width, 4):
            return
        # Write RGBA data and send - needs contiguous uint8 array
        data = np.ascontiguousarray(rgba_array, dtype=np.uint8).ravel()
        self.sender.write_video(data)


def list_audio_devices():
    """List available audio devices"""
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        inputs = dev['max_input_channels']
        if inputs > 0:
            marker = " <-- loopback" if "blackhole" in dev['name'].lower() else ""
            print(f"  [{i}] {dev['name']} ({inputs} ch){marker}")
    print()
    return devices


def main():
    parser = argparse.ArgumentParser(description="Whisper Syphon - Real-time lyrics to NDI")
    parser.add_argument('-d', '--device', type=int, help='Audio device index')
    parser.add_argument('--hf-repo', default="kyutai/stt-1b-en_fr-mlx", help='HuggingFace model repo')
    parser.add_argument('--vad', action='store_true', help='Enable Voice Activity Detection')
    parser.add_argument('--width', type=int, default=1920, help='Output width')
    parser.add_argument('--height', type=int, default=1080, help='Output height')
    parser.add_argument('--fps', type=int, default=30, help='Output FPS')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    parser.add_argument('--max-steps', type=int, default=8192, help='Max generation steps')
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    print("Whisper Syphon (Moshi STT + NDI)")
    print("=" * 35)

    # List devices
    devices = list_audio_devices()

    # Find BlackHole device for system audio
    if args.device is None:
        for i, dev in enumerate(devices):
            if "blackhole" in dev['name'].lower() and dev['max_input_channels'] > 0:
                args.device = i
                print(f"Auto-selected BlackHole device: [{i}] {dev['name']}")
                break
        if args.device is None:
            print("No BlackHole device found. Install BlackHole for system audio capture.")
            print("Or specify a microphone with -d <device_index>")
            args.device = sd.default.device[0]
            print(f"Using default input device: {args.device}")

    # Load Moshi model
    print(f"\nLoading model: {args.hf_repo}", flush=True)
    print("(This may take a minute on first run...)", flush=True)

    print("  Downloading config...", flush=True)
    lm_config_path = hf_hub_download(args.hf_repo, "config.json")
    with open(lm_config_path, "r") as f:
        lm_config_dict = json.load(f)

    print("  Downloading audio tokenizer...", flush=True)
    mimi_weights = hf_hub_download(args.hf_repo, lm_config_dict["mimi_name"])
    moshi_name = lm_config_dict.get("moshi_name", "model.safetensors")
    print("  Downloading model weights...", flush=True)
    moshi_weights = hf_hub_download(args.hf_repo, moshi_name)
    print("  Downloading text tokenizer...", flush=True)
    tokenizer_path = hf_hub_download(args.hf_repo, lm_config_dict["tokenizer_name"])

    lm_config = models.LmConfig.from_config_dict(lm_config_dict)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    if moshi_weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)

    print(f"  Loading model weights into MLX...", flush=True)
    if args.hf_repo.endswith("-candle"):
        model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
    else:
        model.load_weights(moshi_weights, strict=True)
    print("  Done!", flush=True)

    print(f"  Loading text tokenizer...", flush=True)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    print(f"  Loading audio tokenizer...", flush=True)
    generated_codebooks = lm_config.generated_codebooks
    other_codebooks = lm_config.other_codebooks
    mimi_codebooks = max(generated_codebooks, other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)

    print("  Warming up model...", flush=True)
    model.warmup()

    gen = models.LmGen(
        model=model,
        max_steps=args.max_steps,
        text_sampler=utils.Sampler(top_k=25, temp=0),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        check=False,
    )

    # Initialize NDI and renderer
    renderer = TextRenderer(args.width, args.height)
    ndi = NDIOutput("Whisper Lyrics", args.width, args.height, args.fps)
    ndi.start()

    # Start render thread
    render_running = True
    def render_loop():
        frame_time = 1.0 / args.fps
        while render_running:
            start = time.time()
            frame = renderer.render()
            ndi.send_frame(frame)
            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    render_thread = threading.Thread(target=render_loop, daemon=True)
    render_thread.start()

    # Audio processing
    block_queue = queue.Queue()

    def audio_callback(indata, _frames, _time, _status):
        block_queue.put(indata.copy())

    print(f"\nStarting audio capture from device {args.device}...")
    print("Look for 'Whisper Lyrics' in Resolume's NDI sources.")
    print("Press Ctrl+C to quit.\n")

    try:
        with sd.InputStream(
            device=args.device,
            channels=1,
            dtype="float32",
            samplerate=24000,
            blocksize=1920,  # 80ms at 24kHz
            callback=audio_callback,
        ):
            last_print_was_vad = False
            while True:
                block = block_queue.get()
                block = block[None, :, 0]

                # Encode audio to tokens
                other_audio_tokens = audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, :other_codebooks
                ]

                # Generate text
                if args.vad:
                    text_token, vad_heads = gen.step_with_extra_heads(other_audio_tokens[0])
                    if vad_heads:
                        pr_vad = vad_heads[2][0, 0, 0].item()
                        if pr_vad > 0.5 and not last_print_was_vad:
                            print(" [pause]")
                            last_print_was_vad = True
                else:
                    text_token = gen.step(other_audio_tokens[0])

                text_token = text_token[0].item()

                # Decode and display text
                if text_token not in (0, 3):  # Skip special tokens
                    text = text_tokenizer.id_to_piece(text_token)
                    text = text.replace("▁", " ")
                    print(text, end="", flush=True)
                    renderer.add_text(text)
                    last_print_was_vad = False

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        render_running = False
        ndi.stop()


if __name__ == "__main__":
    main()
