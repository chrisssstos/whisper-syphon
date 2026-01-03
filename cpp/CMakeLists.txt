cmake_minimum_required(VERSION 3.20)
project(whisper_syphon VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_MACOS "Build for macOS with ScreenCaptureKit and Syphon" ON)
option(BUILD_WINDOWS "Build for Windows with WASAPI" OFF)

# Disable Metal backend (has compile issues on recent Xcode)
# CPU with Accelerate is still fast on Apple Silicon
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS ON CACHE BOOL "" FORCE)
set(WHISPER_COREML OFF CACHE BOOL "" FORCE)

# whisper.cpp - fetch and build (pinned to stable release)
include(FetchContent)
FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp
    GIT_TAG v1.7.2
)
FetchContent_MakeAvailable(whisper)

# Common sources
set(COMMON_SOURCES
    src/main.cpp
    src/app.cpp
    src/transcription/transcriber.cpp
)

# Platform-specific renderer and GUI
if(APPLE)
    list(APPEND COMMON_SOURCES src/display/renderer.mm)
    list(APPEND COMMON_SOURCES src/gui/gui.mm)
else()
    list(APPEND COMMON_SOURCES src/display/renderer.cpp)
endif()

# Platform-specific sources
if(APPLE AND BUILD_MACOS)
    set(PLATFORM_SOURCES
        src/audio/audio_capture_mac.mm
    )

    find_library(SCREENCAPTUREKIT_FRAMEWORK ScreenCaptureKit)
    find_library(COREMEDIA_FRAMEWORK CoreMedia)
    find_library(COREVIDEO_FRAMEWORK CoreVideo)
    find_library(COREGRAPHICS_FRAMEWORK CoreGraphics)
    find_library(CORETEXT_FRAMEWORK CoreText)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(APPKIT_FRAMEWORK AppKit)
    find_library(OPENGL_FRAMEWORK OpenGL)
    find_library(ACCELERATE_FRAMEWORK Accelerate)

    set(PLATFORM_LIBS
        ${SCREENCAPTUREKIT_FRAMEWORK}
        ${COREMEDIA_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
        ${COREGRAPHICS_FRAMEWORK}
        ${CORETEXT_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${APPKIT_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
    )

    set(PLATFORM_DEFINES PLATFORM_MACOS)

    # Syphon is optional - requires Xcode to build from source
    set(SYPHON_FRAMEWORK ${CMAKE_SOURCE_DIR}/external/Syphon.framework)
    if(EXISTS ${SYPHON_FRAMEWORK}/Syphon AND EXISTS ${SYPHON_FRAMEWORK}/Headers)
        message(STATUS "Syphon framework found - enabling Syphon output")
        list(APPEND PLATFORM_SOURCES src/output/syphon_output.mm)
        list(APPEND PLATFORM_LIBS ${OPENGL_FRAMEWORK} ${SYPHON_FRAMEWORK})
        list(APPEND PLATFORM_DEFINES HAVE_SYPHON)
    else()
        message(STATUS "Syphon framework not found - Syphon output disabled")
        message(STATUS "To enable: install Xcode and build Syphon from external/Syphon-Framework")
    endif()

elseif(WIN32 AND BUILD_WINDOWS)
    set(PLATFORM_SOURCES
        src/audio/audio_capture_win.cpp
    )
    set(PLATFORM_LIBS
        ole32
        winmm
    )
    set(PLATFORM_DEFINES PLATFORM_WINDOWS)
endif()

# Main executable
add_executable(whisper_syphon
    ${COMMON_SOURCES}
    ${PLATFORM_SOURCES}
)

target_include_directories(whisper_syphon PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${whisper_SOURCE_DIR}
)

target_link_libraries(whisper_syphon PRIVATE
    whisper
    ${PLATFORM_LIBS}
)

target_compile_definitions(whisper_syphon PRIVATE
    ${PLATFORM_DEFINES}
)

# macOS bundle settings
if(APPLE)
    set_target_properties(whisper_syphon PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/Info.plist
    )
endif()
