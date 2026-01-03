#pragma once

#include "renderer.h"
#include <string>
#include <memory>

namespace ws {

// Syphon server for outputting frames to other apps (Resolume, etc.)
class SyphonOutput {
public:
    SyphonOutput(const std::string& name, int width, int height);
    ~SyphonOutput();

    // Start the Syphon server
    bool start();

    // Stop the server
    void stop();

    // Publish a frame
    void publish_frame(const Frame& frame);

    // Check if running
    bool is_running() const;

    // Get server name
    const std::string& get_name() const { return name_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string name_;
    int width_;
    int height_;
};

} // namespace ws
