#include "app.h"
#include "gui.h"
#include <iostream>
#include <csignal>

static ws::App* g_app = nullptr;
static ws::GUI* g_gui = nullptr;

void signal_handler(int) {
    if (g_app) {
        g_app->quit();
    }
    if (g_gui) {
        g_gui->quit();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Whisper Syphon v0.1.1 (filtered)\n";
    std::cout << "================================\n\n";

    // Parse args
    ws::ModelSize model_size = ws::ModelSize::Base;
    bool use_gui = true;
    int device_index = -1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tiny") model_size = ws::ModelSize::Tiny;
        else if (arg == "--base") model_size = ws::ModelSize::Base;
        else if (arg == "--small") model_size = ws::ModelSize::Small;
        else if (arg == "--medium") model_size = ws::ModelSize::Medium;
        else if (arg == "--large") model_size = ws::ModelSize::Large;
        else if (arg == "--cli" || arg == "--no-gui") use_gui = false;
        else if (arg == "-d" && i + 1 < argc) {
            device_index = std::stoi(argv[++i]);
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: whisper_syphon [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --tiny/base/small/medium/large  Model size (default: base)\n";
            std::cout << "  --cli, --no-gui                 Run without GUI\n";
            std::cout << "  -d <index>                      Audio device index (CLI mode)\n";
            std::cout << "  -h, --help                      Show this help\n";
            return 0;
        }
    }

    // Create app
    ws::App app;
    g_app = &app;

    // Setup signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize
    std::cout << "Loading Whisper model...\n";
    if (!app.init(model_size)) {
        std::cerr << "Failed to initialize\n";
        return 1;
    }

    if (use_gui) {
        // GUI mode
        ws::GUI gui;
        g_gui = &gui;

        if (!gui.init(&app)) {
            std::cerr << "Failed to initialize GUI\n";
            return 1;
        }

        // Set up word callback to update GUI
        app.set_word_callback([&gui](const std::string& word) {
            gui.update_transcription(word);
        });

        // Run GUI (blocking)
        gui.run();

    } else {
        // CLI mode
        auto devices = app.get_audio_devices();
        if (devices.empty()) {
            std::cerr << "No audio devices found\n";
            return 1;
        }

        std::cout << "\nAvailable audio devices:\n";
        int loopback_device = -1;
        for (const auto& dev : devices) {
            std::cout << "  [" << dev.index << "] " << dev.name;
            if (dev.is_loopback) {
                std::cout << " (loopback)";
                if (loopback_device < 0) loopback_device = dev.index;
            }
            std::cout << "\n";
        }

        if (device_index < 0) {
            device_index = (loopback_device >= 0) ? loopback_device : devices[0].index;
            std::cout << "\nUsing device: " << device_index << "\n";
        }

        std::cout << "\nStarting capture...\n";
        if (!app.start(device_index)) {
            std::cerr << "Failed to start capture\n";
            return 1;
        }

        std::cout << "Running. Press Ctrl+C to quit.\n\n";
        app.run();

        std::cout << "\nStopping...\n";
        app.stop();
    }

    return 0;
}
