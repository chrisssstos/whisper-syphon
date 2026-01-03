#!/bin/bash
# Setup script for Whisper Syphon (Moshi + NDI)

set -e

echo "Whisper Syphon Setup"
echo "===================="

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install BlackHole for system audio capture
echo ""
echo "Installing BlackHole 2ch for system audio capture..."
echo "(This requires your password)"
brew install --cask blackhole-2ch

echo ""
echo "IMPORTANT: After installation, reboot your Mac!"
echo ""
echo "Then set up a Multi-Output Device in Audio MIDI Setup:"
echo "1. Open Audio MIDI Setup (search in Spotlight)"
echo "2. Click + and choose 'Create Multi-Output Device'"
echo "3. Check both 'MacBook Pro Speakers' and 'BlackHole 2ch'"
echo "4. Right-click the Multi-Output Device -> 'Use This Device For Sound Output'"
echo ""
echo "This routes audio to both speakers AND BlackHole (for capture)."

# Create/update virtual environment
echo ""
echo "Setting up Python environment..."
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    uv venv --python 3.11 .venv
fi

source .venv/bin/activate
uv pip install moshi_mlx cyndilib numpy Pillow sounddevice

echo ""
echo "Setup complete!"
echo ""
echo "To run:"
echo "  cd $(pwd)"
echo "  source .venv/bin/activate"
echo "  python whisper_syphon.py"
echo ""
echo "Options:"
echo "  -d <device>    Audio device index (auto-detects BlackHole)"
echo "  --vad          Enable voice activity detection"
echo "  --list-devices List available audio devices"
