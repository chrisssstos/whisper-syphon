#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace ws {

// RGBA frame buffer
struct Frame {
    std::vector<uint8_t> data;
    int width;
    int height;
};

// Text renderer using platform-native APIs
class Renderer {
public:
    Renderer(int width = 1920, int height = 1080);
    ~Renderer();

    // Render text to frame
    Frame render(const std::string& text);

    // Get black frame
    Frame get_black_frame();

    // Settings
    void set_font_size(float size);
    void set_font_name(const std::string& name);
    void set_text_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
    void set_background_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    int width_;
    int height_;
    float font_size_ = 120.0f;
    std::string font_name_ = "Helvetica Neue";
    uint8_t text_color_[4] = {255, 255, 255, 255};
    uint8_t bg_color_[4] = {0, 0, 0, 255};
};

} // namespace ws
