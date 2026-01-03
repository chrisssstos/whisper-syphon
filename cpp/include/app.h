#pragma once

#include "audio_capture.h"
#include "transcriber.h"
#include "renderer.h"

#ifdef HAVE_SYPHON
#include "syphon_output.h"
#endif

#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include <mutex>

namespace ws {

class App {
public:
    static constexpr int WIDTH = 1920;
    static constexpr int HEIGHT = 1080;
    static constexpr int FPS = 30;

    App();
    ~App();

    // Initialize all components
    bool init(ModelSize model_size = ModelSize::Base);

    // Get available audio devices
    std::vector<AudioDevice> get_audio_devices();

    // Start processing with given device
    bool start(int device_index);

    // Stop processing
    void stop();

    // Run main loop (blocking)
    void run();

    // Request quit
    void quit();

    // Get current word
    std::string get_current_word() const;

    // Set callback for new words (for GUI)
    using WordCallback = std::function<void(const std::string&)>;
    void set_word_callback(WordCallback callback);

private:
    void on_word(const Word& word);
    void render_loop();

    std::unique_ptr<AudioCapture> audio_;
    std::unique_ptr<Transcriber> transcriber_;
    std::unique_ptr<Renderer> renderer_;

#ifdef HAVE_SYPHON
    std::unique_ptr<SyphonOutput> syphon_;
#endif

    std::atomic<bool> running_{false};
    std::thread render_thread_;

    mutable std::mutex word_mutex_;
    std::string current_word_;
    WordCallback word_callback_;
};

} // namespace ws
