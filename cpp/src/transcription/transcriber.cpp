#include "transcriber.h"
#include "whisper.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace ws {

struct Transcriber::WhisperContext {
    whisper_context* ctx = nullptr;
    whisper_full_params params;

    ~WhisperContext() {
        if (ctx) {
            whisper_free(ctx);
        }
    }
};

Transcriber::Transcriber() : ctx_(std::make_unique<WhisperContext>()) {
    // Setup default params
    ctx_->params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    ctx_->params.print_realtime = false;
    ctx_->params.print_progress = false;
    ctx_->params.print_timestamps = false;
    ctx_->params.print_special = false;
    ctx_->params.single_segment = true;
    ctx_->params.max_tokens = 32;
    ctx_->params.language = "en";
    ctx_->params.n_threads = 4;

    // Enable token-level timestamps for word output
    ctx_->params.token_timestamps = true;
    ctx_->params.max_len = 1;  // Force shorter segments
}

Transcriber::~Transcriber() {
    stop();
}

std::string Transcriber::model_path(ModelSize size) {
    std::string name;
    switch (size) {
        case ModelSize::Tiny:   name = "ggml-tiny.bin"; break;
        case ModelSize::Base:   name = "ggml-base.bin"; break;
        case ModelSize::Small:  name = "ggml-small.bin"; break;
        case ModelSize::Medium: name = "ggml-medium.bin"; break;
        case ModelSize::Large:  name = "ggml-large.bin"; break;
    }

    // Check common locations
    std::vector<std::string> paths = {
        name,
        "models/" + name,
        "../models/" + name,
        // Absolute path to project models directory
        "/Users/christos/Documents/Gits/whisper-syphon/cpp/models/" + name,
        std::string(getenv("HOME") ? getenv("HOME") : "") + "/.cache/whisper/" + name,
        "/usr/local/share/whisper/" + name
    };

    for (const auto& p : paths) {
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    // Return default path - will fail gracefully in load_model
    return "models/" + name;
}

bool Transcriber::load_model(ModelSize size) {
    std::string path = model_path(size);

    if (!std::filesystem::exists(path)) {
        std::cerr << "Model not found: " << path << "\n";
        std::cerr << "Download from: https://huggingface.co/ggerganov/whisper.cpp\n";
        std::cerr << "Or run: ./models/download-ggml-model.sh base\n";
        return false;
    }

    std::cout << "Loading model: " << path << "\n";
    ctx_->ctx = whisper_init_from_file(path.c_str());

    if (!ctx_->ctx) {
        std::cerr << "Failed to load model\n";
        return false;
    }

    std::cout << "Model loaded successfully\n";
    return true;
}

void Transcriber::set_word_callback(WordCallback callback) {
    word_callback_ = std::move(callback);
}

void Transcriber::push_audio(const float* samples, size_t count) {
    std::lock_guard<std::mutex> lock(audio_mutex_);
    audio_buffer_.insert(audio_buffer_.end(), samples, samples + count);

    // Keep buffer bounded
    if (audio_buffer_.size() > WINDOW_SIZE * 2) {
        audio_buffer_.erase(audio_buffer_.begin(),
                           audio_buffer_.begin() + audio_buffer_.size() - WINDOW_SIZE);
    }
}

bool Transcriber::start() {
    if (running_) return true;
    if (!ctx_->ctx) return false;

    running_ = true;
    process_thread_ = std::thread(&Transcriber::process_loop, this);
    return true;
}

void Transcriber::stop() {
    if (!running_) return;

    running_ = false;
    if (process_thread_.joinable()) {
        process_thread_.join();
    }
}

void Transcriber::process_loop() {
    std::vector<float> window;
    std::string last_text;
    std::vector<std::string> recent_words;  // Track recent words for repetition detection
    const size_t RECENT_HISTORY = 20;

    while (running_) {
        // Get audio window
        {
            std::lock_guard<std::mutex> lock(audio_mutex_);
            if (audio_buffer_.size() < STEP_SIZE) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            // Take window
            size_t window_start = audio_buffer_.size() > WINDOW_SIZE
                                 ? audio_buffer_.size() - WINDOW_SIZE
                                 : 0;
            window.assign(audio_buffer_.begin() + window_start, audio_buffer_.end());

            // Remove processed samples (keep overlap)
            if (audio_buffer_.size() > WINDOW_SIZE) {
                audio_buffer_.erase(audio_buffer_.begin(),
                                   audio_buffer_.begin() + STEP_SIZE);
            }
        }

        if (window.size() < SAMPLE_RATE) {
            continue;  // Need at least 1 second
        }

        // Run whisper
        int result = whisper_full(ctx_->ctx, ctx_->params, window.data(), window.size());
        if (result != 0) {
            continue;
        }

        // Get segments
        int n_segments = whisper_full_n_segments(ctx_->ctx);
        for (int i = 0; i < n_segments; i++) {
            int n_tokens = whisper_full_n_tokens(ctx_->ctx, i);

            for (int j = 0; j < n_tokens; j++) {
                whisper_token_data token = whisper_full_get_token_data(ctx_->ctx, i, j);

                // Skip special tokens
                if (token.id >= whisper_token_eot(ctx_->ctx)) {
                    continue;
                }

                const char* text = whisper_full_get_token_text(ctx_->ctx, i, j);
                if (!text || strlen(text) == 0) continue;

                std::string word(text);

                // Clean up whitespace
                word.erase(0, word.find_first_not_of(" \t\n"));
                word.erase(word.find_last_not_of(" \t\n") + 1);

                if (word.empty()) continue;

                // Filter out non-speech tokens (music, sounds, blank audio, etc.)
                std::string lower = word;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

                // Skip bracketed annotations like [Music], [BLANK_AUDIO], etc.
                if (word.front() == '[' || word.front() == '(' || word.front() == '*') continue;
                if (word.back() == ']' || word.back() == ')' || word.back() == '*') continue;

                // Skip common non-speech markers
                if (lower.find("music") != std::string::npos) continue;
                if (lower.find("blank") != std::string::npos) continue;
                if (lower.find("audio") != std::string::npos) continue;
                if (lower.find("sound") != std::string::npos) continue;
                if (lower.find("noise") != std::string::npos) continue;
                if (lower.find("inaudible") != std::string::npos) continue;
                if (lower.find("applause") != std::string::npos) continue;
                if (lower.find("laughter") != std::string::npos) continue;
                if (lower.find("silence") != std::string::npos) continue;
                if (lower == "_") continue;

                // Skip low-probability tokens (likely hallucinations)
                if (token.p < 0.6f) continue;

                // Avoid duplicates
                if (word == last_text) continue;
                last_text = word;

                // Detect repetitive hallucinations
                int repeat_count = 0;
                for (const auto& recent : recent_words) {
                    if (recent == word) repeat_count++;
                }
                if (repeat_count >= 3) continue;  // Word appeared 3+ times recently - likely hallucination

                // Update recent word history
                recent_words.push_back(word);
                if (recent_words.size() > RECENT_HISTORY) {
                    recent_words.erase(recent_words.begin());
                }

                if (word_callback_) {
                    Word w;
                    w.text = word;
                    w.start_time = static_cast<float>(token.t0) / 100.0f;
                    w.end_time = static_cast<float>(token.t1) / 100.0f;
                    w.probability = token.p;
                    word_callback_(w);
                }
            }
        }
    }
}

} // namespace ws
