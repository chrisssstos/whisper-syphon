#pragma once

#include <functional>
#include <vector>
#include <cstdint>
#include <memory>
#include <string>

namespace ws {

// Audio format for captured samples
struct AudioFormat {
    int sample_rate = 16000;  // Whisper expects 16kHz
    int channels = 1;         // Mono for transcription
    int bits_per_sample = 32; // Float32
};

// Audio device info
struct AudioDevice {
    int index;
    std::string name;
    bool is_loopback;
};

// Callback for audio chunks
// Receives float samples normalized to [-1, 1]
using AudioCallback = std::function<void(const float* samples, size_t count)>;

// Abstract audio capture interface
class AudioCapture {
public:
    virtual ~AudioCapture() = default;

    // Get available audio devices (including loopback)
    virtual std::vector<AudioDevice> get_devices() = 0;

    // Start capturing from device
    virtual bool start(int device_index, AudioCallback callback) = 0;

    // Stop capturing
    virtual void stop() = 0;

    // Check if capturing
    virtual bool is_running() const = 0;

    // Get current format
    virtual AudioFormat get_format() const = 0;
};

// Factory to create platform-specific implementation
std::unique_ptr<AudioCapture> create_audio_capture();

} // namespace ws
