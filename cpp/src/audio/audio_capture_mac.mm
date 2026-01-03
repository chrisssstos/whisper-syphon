#import "audio_capture.h"
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreMedia/CoreMedia.h>
#import <AVFoundation/AVFoundation.h>
#import <Accelerate/Accelerate.h>
#import <atomic>
#import <mutex>

// Forward declare the callback type
using AudioCallbackFn = std::function<void(const float*, size_t)>;

// Delegate to receive audio samples from ScreenCaptureKit
// Must be in global scope for Objective-C
@interface AudioCaptureDelegate : NSObject <SCStreamDelegate, SCStreamOutput>
@property (nonatomic, assign) AudioCallbackFn callback;
@property (nonatomic, assign) int targetSampleRate;
@end

@implementation AudioCaptureDelegate

- (void)stream:(SCStream *)stream didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer ofType:(SCStreamOutputType)type {
    if (type != SCStreamOutputTypeAudio) return;
    if (!self.callback) return;

    // Get audio buffer
    CMBlockBufferRef blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer);
    if (!blockBuffer) return;

    size_t totalLength = 0;
    char* dataPointer = nullptr;
    OSStatus status = CMBlockBufferGetDataPointer(blockBuffer, 0, nullptr, &totalLength, &dataPointer);
    if (status != kCMBlockBufferNoErr || !dataPointer) return;

    // Get format
    CMFormatDescriptionRef formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer);
    const AudioStreamBasicDescription* asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc);
    if (!asbd) return;

    // Convert to float samples
    size_t sampleCount = totalLength / (asbd->mBitsPerChannel / 8) / asbd->mChannelsPerFrame;
    std::vector<float> samples(sampleCount);

    if (asbd->mFormatFlags & kAudioFormatFlagIsFloat) {
        // Already float - copy and mix to mono
        const float* floatData = reinterpret_cast<const float*>(dataPointer);
        if (asbd->mChannelsPerFrame == 1) {
            memcpy(samples.data(), floatData, sampleCount * sizeof(float));
        } else {
            // Mix to mono
            for (size_t i = 0; i < sampleCount; i++) {
                float sum = 0;
                for (UInt32 ch = 0; ch < asbd->mChannelsPerFrame; ch++) {
                    sum += floatData[i * asbd->mChannelsPerFrame + ch];
                }
                samples[i] = sum / asbd->mChannelsPerFrame;
            }
        }
    } else if (asbd->mBitsPerChannel == 16) {
        // 16-bit int to float
        const int16_t* intData = reinterpret_cast<const int16_t*>(dataPointer);
        if (asbd->mChannelsPerFrame == 1) {
            vDSP_vflt16(intData, 1, samples.data(), 1, sampleCount);
            float scale = 1.0f / 32768.0f;
            vDSP_vsmul(samples.data(), 1, &scale, samples.data(), 1, sampleCount);
        } else {
            // Mix to mono
            for (size_t i = 0; i < sampleCount; i++) {
                float sum = 0;
                for (UInt32 ch = 0; ch < asbd->mChannelsPerFrame; ch++) {
                    sum += intData[i * asbd->mChannelsPerFrame + ch] / 32768.0f;
                }
                samples[i] = sum / asbd->mChannelsPerFrame;
            }
        }
    }

    // Resample to 16kHz if needed
    if (static_cast<int>(asbd->mSampleRate) != self.targetSampleRate) {
        double ratio = static_cast<double>(self.targetSampleRate) / asbd->mSampleRate;
        size_t newCount = static_cast<size_t>(sampleCount * ratio);
        std::vector<float> resampled(newCount);

        // Simple linear interpolation
        for (size_t i = 0; i < newCount; i++) {
            double srcIdx = i / ratio;
            size_t idx0 = static_cast<size_t>(srcIdx);
            size_t idx1 = std::min(idx0 + 1, sampleCount - 1);
            double frac = srcIdx - idx0;
            resampled[i] = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
        }

        self.callback(resampled.data(), resampled.size());
    } else {
        self.callback(samples.data(), samples.size());
    }
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
    if (error) {
        NSLog(@"SCStream stopped with error: %@", [error localizedDescription]);
    }
}

@end


namespace ws {

class AudioCaptureMac : public AudioCapture {
public:
    AudioCaptureMac() : delegate_([[AudioCaptureDelegate alloc] init]) {
        delegate_.targetSampleRate = format_.sample_rate;
    }

    ~AudioCaptureMac() override {
        stop();
    }

    std::vector<AudioDevice> get_devices() override {
        std::vector<AudioDevice> devices;

        // ScreenCaptureKit provides system audio - present as a virtual loopback device
        devices.push_back({0, "System Audio (ScreenCaptureKit)", true});

        return devices;
    }

    bool start(int device_index, AudioCallback callback) override {
        if (running_) return true;

        delegate_.callback = callback;

        // Use dispatch semaphore to wait for async setup
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block BOOL setupSuccess = NO;
        __block NSString* errorMessage = nil;

        // Capture self members we need inside the block
        AudioCaptureDelegate* delegate = delegate_;
        __block SCStream* createdStream = nil;

        [SCShareableContent getShareableContentExcludingDesktopWindows:YES
                                                   onScreenWindowsOnly:NO
                                                     completionHandler:^(SCShareableContent* content, NSError* error) {
            // All setup must happen inside this block while objects are valid
            @autoreleasepool {
                if (error) {
                    errorMessage = [NSString stringWithFormat:@"Failed to get shareable content: %@", [error localizedDescription]];
                    dispatch_semaphore_signal(sem);
                    return;
                }

                if (!content || content.displays.count == 0) {
                    errorMessage = content ? @"No displays found" : @"Content is nil";
                    dispatch_semaphore_signal(sem);
                    return;
                }

                // Create filter with display - objects are valid here
                SCDisplay* display = content.displays[0];
                SCContentFilter* filter = [[SCContentFilter alloc] initWithDisplay:display
                                                                  excludingWindows:@[]];

                // Configure stream for audio capture
                SCStreamConfiguration* config = [[SCStreamConfiguration alloc] init];
                config.capturesAudio = YES;
                config.excludesCurrentProcessAudio = YES;
                config.channelCount = 2;
                config.sampleRate = 48000;  // Will resample to 16kHz

                // Minimal video config (required but we won't use it)
                config.width = 2;
                config.height = 2;
                config.minimumFrameInterval = CMTimeMake(1, 1);  // 1 fps
                config.queueDepth = 3;

                // Create stream
                createdStream = [[SCStream alloc] initWithFilter:filter
                                                   configuration:config
                                                        delegate:delegate];

                NSError* addOutputError = nil;

                // Add video output (required by ScreenCaptureKit even for audio-only)
                [createdStream addStreamOutput:delegate
                                          type:SCStreamOutputTypeScreen
                              sampleHandlerQueue:dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0)
                                         error:&addOutputError];

                if (addOutputError) {
                    NSLog(@"Warning: Failed to add video output: %@", [addOutputError localizedDescription]);
                }

                // Add audio output
                addOutputError = nil;
                [createdStream addStreamOutput:delegate
                                          type:SCStreamOutputTypeAudio
                              sampleHandlerQueue:dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0)
                                         error:&addOutputError];

                if (addOutputError) {
                    errorMessage = [NSString stringWithFormat:@"Failed to add audio output: %@", [addOutputError localizedDescription]];
                    createdStream = nil;
                    dispatch_semaphore_signal(sem);
                    return;
                }

                // Start capture
                [createdStream startCaptureWithCompletionHandler:^(NSError* startError) {
                    if (startError) {
                        errorMessage = [NSString stringWithFormat:@"Failed to start capture: %@", [startError localizedDescription]];
                        createdStream = nil;
                    } else {
                        setupSuccess = YES;
                    }
                    dispatch_semaphore_signal(sem);
                }];
            }
        }];

        // Wait for setup to complete
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        if (!setupSuccess) {
            if (errorMessage) {
                NSLog(@"%@", errorMessage);
            }
            return false;
        }

        stream_ = createdStream;
        running_ = true;
        NSLog(@"ScreenCaptureKit audio capture started");
        return true;
    }

    void stop() override {
        if (!running_) return;

        running_ = false;

        if (stream_) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [stream_ stopCaptureWithCompletionHandler:^(NSError* error) {
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
            stream_ = nil;
        }

        NSLog(@"ScreenCaptureKit audio capture stopped");
    }

    bool is_running() const override { return running_; }

    AudioFormat get_format() const override { return format_; }

private:
    AudioCaptureDelegate* delegate_ = nil;
    SCStream* stream_ = nil;
    std::atomic<bool> running_{false};
    AudioFormat format_{16000, 1, 32};
};

std::unique_ptr<AudioCapture> create_audio_capture() {
    if (@available(macOS 13.0, *)) {
        return std::make_unique<AudioCaptureMac>();
    }
    NSLog(@"ScreenCaptureKit audio requires macOS 13.0+");
    return nullptr;
}

} // namespace ws
