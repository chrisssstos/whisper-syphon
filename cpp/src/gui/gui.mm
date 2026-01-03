#import "gui.h"
#import <Cocoa/Cocoa.h>
#import <atomic>

// Forward declaration
namespace ws { class App; }

// AppDelegate
@interface WhisperAppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property (nonatomic) ws::App* app;
@property (nonatomic, strong) NSWindow* window;
@property (nonatomic, strong) NSPopUpButton* sourceDropdown;
@property (nonatomic, strong) NSTextField* transcriptionLabel;
@property (nonatomic, strong) NSTextField* syphonStatusLabel;
@property (nonatomic, strong) NSButton* startStopButton;
@property (nonatomic, strong) NSTextField* infoLabel;
@property (nonatomic) BOOL isRunning;
@property (nonatomic) std::atomic<bool>* shouldQuit;
@end

@implementation WhisperAppDelegate

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    [self setupWindow];
    [self populateDevices];
}

- (void)setupWindow {
    // Create window
    NSRect frame = NSMakeRect(100, 100, 600, 400);
    NSWindowStyleMask style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                              NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:style
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    [self.window setTitle:@"Whisper Syphon"];
    [self.window setDelegate:self];
    [self.window setMinSize:NSMakeSize(400, 300)];

    // Create content view with dark background
    NSView* contentView = [[NSView alloc] initWithFrame:frame];
    [self.window setContentView:contentView];

    // Title label
    NSTextField* titleLabel = [self createLabelWithText:@"Whisper Syphon" fontSize:24 bold:YES];
    titleLabel.frame = NSMakeRect(20, frame.size.height - 60, 560, 30);
    [contentView addSubview:titleLabel];

    // Source label
    NSTextField* sourceLabel = [self createLabelWithText:@"Audio Source:" fontSize:14 bold:NO];
    sourceLabel.frame = NSMakeRect(20, frame.size.height - 100, 120, 25);
    [contentView addSubview:sourceLabel];

    // Source dropdown
    self.sourceDropdown = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(140, frame.size.height - 102, 340, 28) pullsDown:NO];
    [contentView addSubview:self.sourceDropdown];

    // Start/Stop button
    self.startStopButton = [[NSButton alloc] initWithFrame:NSMakeRect(490, frame.size.height - 102, 90, 28)];
    [self.startStopButton setTitle:@"Start"];
    [self.startStopButton setBezelStyle:NSBezelStyleRounded];
    [self.startStopButton setTarget:self];
    [self.startStopButton setAction:@selector(toggleCapture:)];
    [contentView addSubview:self.startStopButton];

    // Transcription box label
    NSTextField* transLabel = [self createLabelWithText:@"Live Transcription:" fontSize:14 bold:NO];
    transLabel.frame = NSMakeRect(20, frame.size.height - 140, 200, 20);
    [contentView addSubview:transLabel];

    // Transcription display (scrollable text area)
    NSScrollView* scrollView = [[NSScrollView alloc] initWithFrame:NSMakeRect(20, 80, 560, frame.size.height - 170)];
    scrollView.hasVerticalScroller = YES;
    scrollView.hasHorizontalScroller = NO;
    scrollView.autohidesScrollers = YES;
    scrollView.borderType = NSBezelBorder;
    scrollView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    self.transcriptionLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 540, frame.size.height - 170)];
    self.transcriptionLabel.editable = NO;
    self.transcriptionLabel.selectable = YES;
    self.transcriptionLabel.bordered = NO;
    self.transcriptionLabel.drawsBackground = YES;
    self.transcriptionLabel.backgroundColor = [NSColor colorWithWhite:0.15 alpha:1.0];
    self.transcriptionLabel.textColor = [NSColor whiteColor];
    self.transcriptionLabel.font = [NSFont systemFontOfSize:16];
    self.transcriptionLabel.stringValue = @"Waiting for audio...";
    self.transcriptionLabel.alignment = NSTextAlignmentLeft;
    self.transcriptionLabel.lineBreakMode = NSLineBreakByWordWrapping;
    self.transcriptionLabel.usesSingleLineMode = NO;
    self.transcriptionLabel.cell.wraps = YES;
    self.transcriptionLabel.cell.scrollable = NO;

    scrollView.documentView = self.transcriptionLabel;
    [contentView addSubview:scrollView];

    // Status bar at bottom
    NSBox* statusBar = [[NSBox alloc] initWithFrame:NSMakeRect(0, 0, frame.size.width, 60)];
    statusBar.boxType = NSBoxCustom;
    statusBar.fillColor = [NSColor colorWithWhite:0.1 alpha:1.0];
    statusBar.borderWidth = 0;
    statusBar.autoresizingMask = NSViewWidthSizable;
    [contentView addSubview:statusBar];

    // Syphon status
    self.syphonStatusLabel = [self createLabelWithText:@"Syphon: Not Started" fontSize:12 bold:NO];
    self.syphonStatusLabel.frame = NSMakeRect(20, 20, 250, 20);
    self.syphonStatusLabel.textColor = [NSColor grayColor];
    [contentView addSubview:self.syphonStatusLabel];

    // Info label
    self.infoLabel = [self createLabelWithText:@"Output: 'Whisper Lyrics' via Syphon" fontSize:12 bold:NO];
    self.infoLabel.frame = NSMakeRect(280, 20, 300, 20);
    self.infoLabel.textColor = [NSColor grayColor];
    self.infoLabel.alignment = NSTextAlignmentRight;
    [contentView addSubview:self.infoLabel];

    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (NSTextField*)createLabelWithText:(NSString*)text fontSize:(CGFloat)size bold:(BOOL)bold {
    NSTextField* label = [[NSTextField alloc] init];
    label.stringValue = text;
    label.editable = NO;
    label.selectable = NO;
    label.bordered = NO;
    label.drawsBackground = NO;
    label.textColor = [NSColor whiteColor];
    label.font = bold ? [NSFont boldSystemFontOfSize:size] : [NSFont systemFontOfSize:size];
    return label;
}

- (void)populateDevices {
    [self.sourceDropdown removeAllItems];

    if (self.app) {
        auto devices = self.app->get_audio_devices();
        for (const auto& dev : devices) {
            NSString* name = [NSString stringWithUTF8String:dev.name.c_str()];
            if (dev.is_loopback) {
                name = [name stringByAppendingString:@" (System Audio)"];
            }
            [self.sourceDropdown addItemWithTitle:name];
            self.sourceDropdown.lastItem.tag = dev.index;
        }
    }

    if (self.sourceDropdown.numberOfItems == 0) {
        [self.sourceDropdown addItemWithTitle:@"No devices found"];
        self.startStopButton.enabled = NO;
    }
}

- (void)toggleCapture:(id)sender {
    if (self.isRunning) {
        // Stop
        if (self.app) {
            self.app->stop();
        }
        self.isRunning = NO;
        [self.startStopButton setTitle:@"Start"];
        self.syphonStatusLabel.stringValue = @"Syphon: Stopped";
        self.syphonStatusLabel.textColor = [NSColor grayColor];
        self.sourceDropdown.enabled = YES;
    } else {
        // Start
        if (self.app) {
            NSInteger deviceIndex = self.sourceDropdown.selectedItem.tag;
            if (self.app->start((int)deviceIndex)) {
                self.isRunning = YES;
                [self.startStopButton setTitle:@"Stop"];
                self.syphonStatusLabel.stringValue = @"Syphon: Running - 'Whisper Lyrics'";
                self.syphonStatusLabel.textColor = [NSColor greenColor];
                self.sourceDropdown.enabled = NO;
                self.transcriptionLabel.stringValue = @"";
            } else {
                NSAlert* alert = [[NSAlert alloc] init];
                alert.messageText = @"Failed to Start";
                alert.informativeText = @"Could not start audio capture. Check permissions in System Settings > Privacy & Security > Screen Recording.";
                [alert addButtonWithTitle:@"OK"];
                [alert runModal];
            }
        }
    }
}

- (void)updateTranscription:(NSString*)text {
    dispatch_async(dispatch_get_main_queue(), ^{
        // Append text
        NSString* current = self.transcriptionLabel.stringValue;
        if ([current isEqualToString:@"Waiting for audio..."] || [current isEqualToString:@""]) {
            self.transcriptionLabel.stringValue = text;
        } else {
            self.transcriptionLabel.stringValue = [current stringByAppendingFormat:@" %@", text];
        }

        // Auto-scroll to bottom
        NSScrollView* scrollView = (NSScrollView*)self.transcriptionLabel.superview.superview;
        if ([scrollView isKindOfClass:[NSScrollView class]]) {
            NSPoint newScrollOrigin = NSMakePoint(0, NSMaxY(self.transcriptionLabel.frame) - NSHeight(scrollView.contentView.bounds));
            [scrollView.contentView scrollToPoint:newScrollOrigin];
            [scrollView reflectScrolledClipView:scrollView.contentView];
        }
    });
}

- (BOOL)windowShouldClose:(NSWindow*)sender {
    if (self.isRunning && self.app) {
        self.app->stop();
    }
    if (self.shouldQuit) {
        *self.shouldQuit = true;
    }
    [NSApp terminate:nil];
    return YES;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)app {
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification {
    if (self.isRunning && self.app) {
        self.app->stop();
    }
}

@end


namespace ws {

struct GUI::Impl {
    WhisperAppDelegate* appDelegate = nil;
    std::atomic<bool> shouldQuit{false};
};

GUI::GUI() : impl_(std::make_unique<Impl>()) {}

GUI::~GUI() {
    quit();
}

bool GUI::init(App* app) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        // Create app delegate
        impl_->appDelegate = [[WhisperAppDelegate alloc] init];
        impl_->appDelegate.app = app;
        impl_->appDelegate.shouldQuit = &impl_->shouldQuit;
        [NSApp setDelegate:impl_->appDelegate];

        // Set up word callback to update GUI
        if (app) {
            // We'll update this from the run loop
        }

        return true;
    }
}

void GUI::run() {
    @autoreleasepool {
        [NSApp run];
    }
}

void GUI::update_transcription(const std::string& text) {
    if (impl_->appDelegate) {
        NSString* nsText = [NSString stringWithUTF8String:text.c_str()];
        [impl_->appDelegate updateTranscription:nsText];
    }
}

void GUI::update_syphon_status(bool connected) {
    // Status is updated in toggleCapture
}

void GUI::quit() {
    impl_->shouldQuit = true;
    dispatch_async(dispatch_get_main_queue(), ^{
        [NSApp terminate:nil];
    });
}

} // namespace ws
