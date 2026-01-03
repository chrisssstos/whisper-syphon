#!/bin/bash
# Run Whisper Syphon
cd "$(dirname "$0")"
source .venv/bin/activate
exec python whisper_syphon.py "$@"
