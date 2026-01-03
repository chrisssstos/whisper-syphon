# Whisper Syphon (C++)

Real-time lyrics/speech transcription with Syphon output for VJs.

## Requirements

- macOS 13.0+ (for ScreenCaptureKit audio capture)
- CMake 3.20+
- Xcode Command Line Tools

## Dependencies

### Syphon Framework

Download Syphon.framework and place it in `external/`:

```bash
cd external
curl -L https://github.com/Syphon/Syphon-Framework/releases/download/5/Syphon.SDK.5.zip -o syphon.zip
unzip syphon.zip
cp -r "Syphon SDK 5/Syphon.framework" .
rm -rf "Syphon SDK 5" syphon.zip
```

### Whisper Model

Download a model from [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp):

```bash
mkdir -p models
cd models
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -o ggml-base.bin
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j8
```

## Run

```bash
./whisper_syphon
```

Options:
- `--tiny/base/small/medium/large` - Model size (default: base)
- `-d <index>` - Audio device index

## Usage

1. Run the app - it will auto-select the system audio loopback
2. Open Resolume (or any Syphon client)
3. Add a Syphon input source
4. Select "Whisper Lyrics" server
5. Play music with vocals - lyrics appear in real-time

## Architecture

```
src/
├── main.cpp              # Entry point, CLI
├── app.cpp               # Main application
├── audio/
│   └── audio_capture_mac.mm   # ScreenCaptureKit audio
├── transcription/
│   └── transcriber.cpp   # whisper.cpp wrapper
├── display/
│   └── renderer.cpp      # CoreText rendering
└── output/
    └── syphon_output.mm  # Syphon server
```

## Permissions

On first run, macOS will ask for:
- **Screen Recording** - Required for system audio capture (ScreenCaptureKit)

Grant this in System Preferences → Privacy & Security → Screen Recording.

## License

MIT
