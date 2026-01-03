"""Audio device enumeration."""

import sounddevice as sd
from typing import List, Dict, Optional


def get_input_devices() -> List[Dict]:
    """Get list of available input devices.

    Returns:
        List of dicts with 'index', 'name', and 'channels' keys.
    """
    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': dev['name'],
                'channels': dev['max_input_channels'],
                'sample_rate': dev['default_samplerate']
            })

    return input_devices


def find_device_by_name(name: str) -> Optional[int]:
    """Find device index by partial name match.

    Args:
        name: Partial device name to search for (case-insensitive).

    Returns:
        Device index if found, None otherwise.
    """
    devices = get_input_devices()
    name_lower = name.lower()

    for dev in devices:
        if name_lower in dev['name'].lower():
            return dev['index']

    return None


def get_default_input_device() -> Optional[int]:
    """Get the default input device index."""
    try:
        return sd.default.device[0]
    except Exception:
        devices = get_input_devices()
        return devices[0]['index'] if devices else None
