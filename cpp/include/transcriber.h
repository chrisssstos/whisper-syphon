#pragma once

#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>

namespace ws {

// Word with timing info
struct Word {
    std::string text;
    float start_time;
    float end_time;
    float probability;
};

// Callback for transcribed words
using WordCallback = std::function<void(const Word& word)>;

// Whisper model sizes
enum class ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large
};

// Transcriber wrapping whisper.cpp
class Transcriber {
public:
    Transcriber();
    ~Transcriber();

    // Load model (downloads if needed)
    bool load_model(ModelSize size = ModelSize::Base);

    // Set callback for word output
    void set_word_callback(WordCallback callback);

    // Push audio samples for processing
    void push_audio(const float* samples, size_t count);

    // Start/stop processing thread
    bool start();
    void stop();

    bool is_running() const { return running_; }

private:
    void process_loop();
    std::string model_path(ModelSize size);

    struct WhisperContext;
    std::unique_ptr<WhisperContext> ctx_;

    WordCallback word_callback_;

    std::atomic<bool> running_{false};
    std::thread process_thread_;

    std::mutex audio_mutex_;
    std::vector<float> audio_buffer_;

    // Sliding window for streaming
    static constexpr size_t SAMPLE_RATE = 16000;
    static constexpr size_t WINDOW_SIZE = SAMPLE_RATE * 4;  // 4 second window
    static constexpr size_t STEP_SIZE = SAMPLE_RATE * 3;    // 3 second step (less overlap)
};

} // namespace ws
