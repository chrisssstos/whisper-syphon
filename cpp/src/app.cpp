#include "app.h"
#include <iostream>
#include <chrono>

namespace ws {

App::App() = default;
App::~App() { stop(); }

bool App::init(ModelSize model_size) {
    // Create components
    audio_ = create_audio_capture();
    if (!audio_) {
        std::cerr << "Failed to create audio capture\n";
        return false;
    }

    transcriber_ = std::make_unique<Transcriber>();
    if (!transcriber_->load_model(model_size)) {
        std::cerr << "Failed to load Whisper model\n";
        return false;
    }

    transcriber_->set_word_callback([this](const Word& w) {
        on_word(w);
    });

    renderer_ = std::make_unique<Renderer>(WIDTH, HEIGHT);

#ifdef HAVE_SYPHON
    syphon_ = std::make_unique<SyphonOutput>("Whisper Lyrics", WIDTH, HEIGHT);
#endif

    return true;
}

std::vector<AudioDevice> App::get_audio_devices() {
    return audio_ ? audio_->get_devices() : std::vector<AudioDevice>{};
}

bool App::start(int device_index) {
    if (running_) return true;

    // Start transcriber
    if (!transcriber_->start()) {
        std::cerr << "Failed to start transcriber\n";
        return false;
    }

    // Start audio capture
    bool audio_ok = audio_->start(device_index, [this](const float* samples, size_t count) {
        transcriber_->push_audio(samples, count);
    });

    if (!audio_ok) {
        std::cerr << "Failed to start audio capture\n";
        transcriber_->stop();
        return false;
    }

#ifdef HAVE_SYPHON
    if (syphon_ && !syphon_->start()) {
        std::cerr << "Warning: Failed to start Syphon server\n";
        // Continue anyway - Syphon is optional
    }
#endif

    running_ = true;

    // Start render thread
    render_thread_ = std::thread(&App::render_loop, this);

    return true;
}

void App::stop() {
    if (!running_) return;

    running_ = false;

    if (render_thread_.joinable()) {
        render_thread_.join();
    }

    if (audio_) audio_->stop();
    if (transcriber_) transcriber_->stop();

#ifdef HAVE_SYPHON
    if (syphon_) syphon_->stop();
#endif
}

void App::run() {
    // Simple main loop - just wait for quit
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void App::quit() {
    running_ = false;
}

std::string App::get_current_word() const {
    std::lock_guard<std::mutex> lock(word_mutex_);
    return current_word_;
}

void App::on_word(const Word& word) {
    {
        std::lock_guard<std::mutex> lock(word_mutex_);
        current_word_ = word.text;
    }
    std::cout << word.text << " " << std::flush;

    // Notify GUI if callback is set
    if (word_callback_) {
        word_callback_(word.text);
    }
}

void App::set_word_callback(WordCallback callback) {
    word_callback_ = callback;
}

void App::render_loop() {
    using namespace std::chrono;
    const auto frame_duration = milliseconds(1000 / FPS);
    std::string last_word;

    while (running_) {
        auto frame_start = steady_clock::now();

        // Get current word
        std::string word = get_current_word();

        // Render frame
        Frame frame;
        if (word != last_word || last_word.empty()) {
            frame = renderer_->render(word);
            last_word = word;
        } else {
            // Reuse last frame if word hasn't changed
            frame = renderer_->render(word);
        }

#ifdef HAVE_SYPHON
        // Publish to Syphon
        if (syphon_ && syphon_->is_running()) {
            syphon_->publish_frame(frame);
        }
#endif

        // Maintain frame rate
        auto elapsed = steady_clock::now() - frame_start;
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }
    }
}

} // namespace ws
