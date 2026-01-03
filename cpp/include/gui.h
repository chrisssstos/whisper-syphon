#pragma once

#include "app.h"
#include <functional>
#include <string>
#include <vector>

namespace ws {

class GUI {
public:
    GUI();
    ~GUI();

    // Initialize GUI with app reference
    bool init(App* app);

    // Run the GUI (blocking - runs the NSApplication run loop)
    void run();

    // Update the transcription display
    void update_transcription(const std::string& text);

    // Update Syphon status
    void update_syphon_status(bool connected);

    // Request quit
    void quit();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ws
