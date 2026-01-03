#include "renderer.h"

#ifdef PLATFORM_MACOS
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <Foundation/Foundation.h>
#endif

#include <cstring>

namespace ws {

struct Renderer::Impl {
#ifdef PLATFORM_MACOS
    CGColorSpaceRef colorSpace = nullptr;
    CGContextRef context = nullptr;
    std::vector<uint8_t> buffer;

    Impl(int width, int height) {
        colorSpace = CGColorSpaceCreateDeviceRGB();
        buffer.resize(width * height * 4);

        context = CGBitmapContextCreate(
            buffer.data(),
            width,
            height,
            8,
            width * 4,
            colorSpace,
            kCGImageAlphaPremultipliedLast
        );
    }

    ~Impl() {
        if (context) CGContextRelease(context);
        if (colorSpace) CGColorSpaceRelease(colorSpace);
    }
#endif
};

Renderer::Renderer(int width, int height)
    : width_(width), height_(height) {
#ifdef PLATFORM_MACOS
    impl_ = std::make_unique<Impl>(width, height);
#endif
}

Renderer::~Renderer() = default;

Frame Renderer::render(const std::string& text) {
    Frame frame;
    frame.width = width_;
    frame.height = height_;
    frame.data.resize(width_ * height_ * 4);

#ifdef PLATFORM_MACOS
    CGContextRef ctx = impl_->context;

    // Clear background
    CGContextSetRGBFillColor(ctx,
        bg_color_[0] / 255.0, bg_color_[1] / 255.0,
        bg_color_[2] / 255.0, bg_color_[3] / 255.0);
    CGContextFillRect(ctx, CGRectMake(0, 0, width_, height_));

    if (!text.empty()) {
        // Create attributed string
        CFStringRef fontName = CFStringCreateWithCString(nullptr, font_name_.c_str(), kCFStringEncodingUTF8);
        CTFontRef font = CTFontCreateWithName(fontName, font_size_, nullptr);
        CFRelease(fontName);

        CGColorRef color = CGColorCreateGenericRGB(
            text_color_[0] / 255.0, text_color_[1] / 255.0,
            text_color_[2] / 255.0, text_color_[3] / 255.0);

        CFStringRef keys[] = { kCTFontAttributeName, kCTForegroundColorAttributeName };
        CFTypeRef values[] = { font, color };
        CFDictionaryRef attrs = CFDictionaryCreate(nullptr,
            (const void**)keys, (const void**)values, 2,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

        CFStringRef str = CFStringCreateWithCString(nullptr, text.c_str(), kCFStringEncodingUTF8);
        CFAttributedStringRef attrStr = CFAttributedStringCreate(nullptr, str, attrs);

        // Create line and measure
        CTLineRef line = CTLineCreateWithAttributedString(attrStr);

        CGFloat ascent, descent, leading;
        double lineWidth = CTLineGetTypographicBounds(line, &ascent, &descent, &leading);

        // Center text
        CGFloat x = (width_ - lineWidth) / 2.0;
        CGFloat y = (height_ - (ascent + descent)) / 2.0 + descent;

        // Draw
        CGContextSetTextPosition(ctx, x, y);
        CTLineDraw(line, ctx);

        // Cleanup
        CFRelease(line);
        CFRelease(attrStr);
        CFRelease(str);
        CFRelease(attrs);
        CFRelease(font);
        CGColorRelease(color);
    }

    // Copy to frame
    memcpy(frame.data.data(), impl_->buffer.data(), frame.data.size());
#else
    // Fallback: just fill with background color
    for (int i = 0; i < width_ * height_; i++) {
        frame.data[i * 4 + 0] = bg_color_[0];
        frame.data[i * 4 + 1] = bg_color_[1];
        frame.data[i * 4 + 2] = bg_color_[2];
        frame.data[i * 4 + 3] = bg_color_[3];
    }
#endif

    return frame;
}

Frame Renderer::get_black_frame() {
    Frame frame;
    frame.width = width_;
    frame.height = height_;
    frame.data.resize(width_ * height_ * 4, 0);
    return frame;
}

void Renderer::set_font_size(float size) { font_size_ = size; }
void Renderer::set_font_name(const std::string& name) { font_name_ = name; }

void Renderer::set_text_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    text_color_[0] = r; text_color_[1] = g;
    text_color_[2] = b; text_color_[3] = a;
}

void Renderer::set_background_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    bg_color_[0] = r; bg_color_[1] = g;
    bg_color_[2] = b; bg_color_[3] = a;
}

} // namespace ws
