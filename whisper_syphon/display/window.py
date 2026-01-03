"""Main application window with tkinter."""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from typing import Callable, List, Dict, Optional
import threading


class MainWindow:
    """Main application window with device selector and preview."""

    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 360

    def __init__(self, devices: List[Dict],
                 on_start: Callable[[int], None],
                 on_stop: Callable[[], None]):
        """Initialize main window.

        Args:
            devices: List of audio input devices.
            on_start: Callback when start is pressed, receives device index.
            on_stop: Callback when stop is pressed.
        """
        self.devices = devices
        self.on_start = on_start
        self.on_stop = on_stop

        self._running = False
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Create window
        self.root = tk.Tk()
        self.root.title("Whisper Syphon")
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(False, False)

        # Main frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a', padx=20, pady=20)
        main_frame.pack()

        # Preview canvas
        self.canvas = tk.Canvas(
            main_frame,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            bg='black',
            highlightthickness=1,
            highlightbackground='#333'
        )
        self.canvas.pack(pady=(0, 15))

        # Current word label (overlay)
        self.word_label = tk.Label(
            main_frame,
            text="Ready",
            font=("Helvetica", 24),
            fg='#666',
            bg='#1a1a1a'
        )
        self.word_label.pack(pady=(0, 15))

        # Controls frame
        controls_frame = tk.Frame(main_frame, bg='#1a1a1a')
        controls_frame.pack(fill='x')

        # Device selector
        device_label = tk.Label(
            controls_frame,
            text="Audio Input:",
            font=("Helvetica", 12),
            fg='white',
            bg='#1a1a1a'
        )
        device_label.pack(side='left', padx=(0, 10))

        self.device_var = tk.StringVar()
        device_names = [d['name'] for d in devices]
        if device_names:
            self.device_var.set(device_names[0])

        self.device_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.device_var,
            values=device_names,
            state='readonly',
            width=30
        )
        self.device_dropdown.pack(side='left', padx=(0, 15))

        # Start/Stop button
        self.start_button = tk.Button(
            controls_frame,
            text="Start",
            font=("Helvetica", 12, "bold"),
            fg='white',
            bg='#2d5a27',
            activebackground='#3d7a37',
            width=10,
            command=self._toggle_running
        )
        self.start_button.pack(side='left')

        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Select audio input and click Start",
            font=("Helvetica", 10),
            fg='#888',
            bg='#1a1a1a'
        )
        self.status_label.pack(pady=(15, 0))

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_running(self) -> None:
        """Toggle between running and stopped states."""
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        """Start processing."""
        device_name = self.device_var.get()
        device_index = None

        for dev in self.devices:
            if dev['name'] == device_name:
                device_index = dev['index']
                break

        if device_index is None:
            self.status_label.config(text="Error: No device selected")
            return

        self._running = True
        self.start_button.config(text="Stop", bg='#8b2020', activebackground='#ab3030')
        self.device_dropdown.config(state='disabled')
        self.status_label.config(text=f"Listening on: {device_name}")
        self.word_label.config(text="Listening...", fg='#888')

        self.on_start(device_index)

    def _stop(self) -> None:
        """Stop processing."""
        self._running = False
        self.start_button.config(text="Start", bg='#2d5a27', activebackground='#3d7a37')
        self.device_dropdown.config(state='readonly')
        self.status_label.config(text="Stopped")
        self.word_label.config(text="Ready", fg='#666')

        self.on_stop()

    def _on_close(self) -> None:
        """Handle window close."""
        if self._running:
            self.on_stop()
        self.root.destroy()

    def update_word(self, word: str) -> None:
        """Update the displayed word.

        Args:
            word: Word to display.
        """
        self.word_label.config(text=word, fg='white')

    def update_preview(self, frame: np.ndarray) -> None:
        """Update the preview canvas.

        Args:
            frame: RGBA numpy array.
        """
        try:
            # Resize for preview
            img = Image.fromarray(frame)
            img = img.resize((self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self._photo = ImageTk.PhotoImage(img)

            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

        except Exception as e:
            print(f"Preview update error: {e}")

    def set_status(self, text: str) -> None:
        """Set status label text."""
        self.status_label.config(text=text)

    def run(self) -> None:
        """Run the tkinter main loop."""
        self.root.mainloop()

    def schedule(self, func: Callable, delay_ms: int = 0) -> None:
        """Schedule a function to run on the main thread.

        Args:
            func: Function to call.
            delay_ms: Delay in milliseconds.
        """
        self.root.after(delay_ms, func)
