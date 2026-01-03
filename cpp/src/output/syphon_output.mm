#import "syphon_output.h"
#import <Syphon/Syphon.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>
#import <AppKit/AppKit.h>
#import <mutex>

namespace ws {

struct SyphonOutput::Impl {
    SyphonServer* server = nil;
    NSOpenGLContext* glContext = nil;
    GLuint texture = 0;
    std::mutex mutex;
    bool running = false;
};

SyphonOutput::SyphonOutput(const std::string& name, int width, int height)
    : impl_(std::make_unique<Impl>())
    , name_(name)
    , width_(width)
    , height_(height) {
}

SyphonOutput::~SyphonOutput() {
    stop();
}

bool SyphonOutput::start() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->running) return true;

    // Create OpenGL context
    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFAAccelerated,
        NSOpenGLPFAColorSize, 24,
        NSOpenGLPFAAlphaSize, 8,
        NSOpenGLPFADoubleBuffer,
        0
    };

    NSOpenGLPixelFormat* pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    if (!pixelFormat) {
        NSLog(@"Failed to create pixel format");
        return false;
    }

    impl_->glContext = [[NSOpenGLContext alloc] initWithFormat:pixelFormat shareContext:nil];
    if (!impl_->glContext) {
        NSLog(@"Failed to create OpenGL context");
        return false;
    }

    [impl_->glContext makeCurrentContext];

    // Create texture
    glGenTextures(1, &impl_->texture);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, impl_->texture);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate texture
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8,
                 width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Create Syphon server
    NSString* serverName = [NSString stringWithUTF8String:name_.c_str()];
    impl_->server = [[SyphonServer alloc] initWithName:serverName
                                               context:impl_->glContext.CGLContextObj
                                               options:nil];

    if (!impl_->server) {
        NSLog(@"Failed to create Syphon server");
        glDeleteTextures(1, &impl_->texture);
        impl_->texture = 0;
        return false;
    }

    impl_->running = true;
    NSLog(@"Syphon server started: %@", serverName);

    return true;
}

void SyphonOutput::stop() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!impl_->running) return;

    impl_->running = false;

    if (impl_->server) {
        [impl_->server stop];
        impl_->server = nil;
    }

    if (impl_->texture) {
        [impl_->glContext makeCurrentContext];
        glDeleteTextures(1, &impl_->texture);
        impl_->texture = 0;
    }

    impl_->glContext = nil;

    NSLog(@"Syphon server stopped");
}

void SyphonOutput::publish_frame(const Frame& frame) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!impl_->running || !impl_->server) return;

    [impl_->glContext makeCurrentContext];

    // Upload frame to texture
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, impl_->texture);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0,
                    frame.width, frame.height,
                    GL_RGBA, GL_UNSIGNED_BYTE, frame.data.data());

    // Publish to Syphon
    [impl_->server publishFrameTexture:impl_->texture
                         textureTarget:GL_TEXTURE_RECTANGLE_ARB
                           imageRegion:NSMakeRect(0, 0, frame.width, frame.height)
                     textureDimensions:NSMakeSize(frame.width, frame.height)
                               flipped:YES];
}

bool SyphonOutput::is_running() const {
    return impl_->running;
}

} // namespace ws
