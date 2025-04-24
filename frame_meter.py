import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
from dataclasses import dataclass
import os


# Define the different states a fighter can be in
class FrameState:
    NEUTRAL = "neutral"
    STARTUP = "startup"
    ACTIVE = "active"
    RECOVERY = "recovery"
    HITSTUN = "hitstun"
    BLOCKSTUN = "blockstun"
    PARRY = "parry"
    DODGE = "dodge"


# Color scheme based on SF6 (RGB format)
STATE_COLORS = {
    FrameState.NEUTRAL: (26, 26, 26),
    FrameState.STARTUP: (67, 243, 241),
    FrameState.ACTIVE: (194, 46, 104),
    FrameState.RECOVERY: (3, 107, 180),
    FrameState.HITSTUN: (246, 245, 57),
    FrameState.BLOCKSTUN: (200, 200, 50),
    FrameState.PARRY: (91, 30, 110),
    FrameState.DODGE: (193, 192, 190),
}


@dataclass
class FighterFrameData:
    frame: int
    state: str


class FrameDataAnnotator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Frame Data Annotator")

        # Video properties
        self.video_path = None
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False

        # Frame data - complete history arrays
        self.fighter1_states = []  # Complete history for all frames
        self.fighter2_states = []  # Complete history for all frames

        # Frame meter window (80 frames)
        self.meter_size = 80
        self.meter_index = 0  # Current position in the circular buffer

        # Initialize meter buffers with neutral states
        self.meter_fighter1_states = [FrameState.NEUTRAL] * self.meter_size
        self.meter_fighter2_states = [FrameState.NEUTRAL] * self.meter_size

        # Add sliding window for write mode
        self.window_start = 0  # Start of sliding window for write mode

        # Add read/write mode
        self.write_mode = False

        # UI elements
        self.setup_ui()

        # Keyboard shortcuts
        self.setup_keyboard_shortcuts()

    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top toolbar
        toolbar = ttk.Frame(main_container)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(toolbar, text="Open Video", command=self.open_video).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(toolbar, text="Save Data", command=self.save_data).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(toolbar, text="Load Data", command=self.load_data).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(toolbar, text="Export Video", command=self.export_video).pack(
            side=tk.LEFT, padx=5
        )

        # Video display area
        self.video_frame = ttk.Frame(main_container)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(self.video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Playback controls
        controls = ttk.Frame(main_container)
        controls.pack(fill=tk.X, pady=10)

        # Add read/write mode toggle after the playback controls
        mode_frame = ttk.Frame(main_container)
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value="Read")
        self.mode_toggle = ttk.Checkbutton(
            mode_frame,
            text="Write Mode",
            variable=self.mode_var,
            onvalue="Write",
            offvalue="Read",
            command=self.toggle_mode,
        )
        self.mode_toggle.pack(side=tk.LEFT, padx=5)

        ttk.Button(controls, text="▮◀", command=self.prev_frame).pack(
            side=tk.LEFT, padx=5
        )
        self.play_button = ttk.Button(controls, text="▶", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="▶▮", command=self.next_frame).pack(
            side=tk.LEFT, padx=5
        )

        self.frame_slider = ttk.Scale(
            controls, from_=0, to=0, orient=tk.HORIZONTAL, command=self.slider_changed
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.frame_label = ttk.Label(controls, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        # Annotation controls
        annotation_frame = ttk.LabelFrame(main_container, text="Frame Data Annotation")
        annotation_frame.pack(fill=tk.X, pady=10)

        # Fighter 1 controls - store the frame for later reference
        self.f1_frame = ttk.Frame(annotation_frame)
        self.f1_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.f1_frame, text="Fighter 1:").pack(side=tk.LEFT)
        self.f1_state = tk.StringVar(value=FrameState.NEUTRAL)
        for state in [
            FrameState.NEUTRAL,
            FrameState.STARTUP,
            FrameState.ACTIVE,
            FrameState.RECOVERY,
            FrameState.HITSTUN,
            FrameState.BLOCKSTUN,
            FrameState.PARRY,
            FrameState.DODGE,
        ]:
            ttk.Radiobutton(
                self.f1_frame,
                text=state,
                value=state,
                variable=self.f1_state,
                command=lambda: self.update_fighter_state(1),
            ).pack(side=tk.LEFT, padx=5)

        # Fighter 2 controls - store the frame for later reference
        self.f2_frame = ttk.Frame(annotation_frame)
        self.f2_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.f2_frame, text="Fighter 2:").pack(side=tk.LEFT)
        self.f2_state = tk.StringVar(value=FrameState.NEUTRAL)
        for state in [
            FrameState.NEUTRAL,
            FrameState.STARTUP,
            FrameState.ACTIVE,
            FrameState.RECOVERY,
            FrameState.HITSTUN,
            FrameState.BLOCKSTUN,
            FrameState.PARRY,
            FrameState.DODGE,
        ]:
            ttk.Radiobutton(
                self.f2_frame,
                text=state,
                value=state,
                variable=self.f2_state,
                command=lambda: self.update_fighter_state(2),
            ).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_container, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def toggle_mode(self):
        mode = self.mode_var.get()
        self.write_mode = mode == "Write"

        # Update the checkbutton text
        self.mode_toggle.configure(text=f"{mode} Mode")

        # Update status bar
        self.status_var.set(f"Switched to {mode} mode")

        # Update radio buttons state based on current mode
        self.update_radio_buttons_state()

        # When switching to read mode, update radio buttons to reflect current frame data
        if not self.write_mode:
            self.f1_state.set(self.fighter1_states[self.current_frame])
            self.f2_state.set(self.fighter2_states[self.current_frame])

        # Update frame to refresh the display with the new mode's visualization
        self.update_frame()

    def update_radio_buttons_state(self):
        state = "normal" if self.write_mode else "disabled"

        # Update all radio buttons in fighter 1 frame
        for widget in self.f1_frame.winfo_children():
            if isinstance(widget, ttk.Radiobutton):
                widget.configure(state=state)

        # Update all radio buttons in fighter 2 frame
        for widget in self.f2_frame.winfo_children():
            if isinstance(widget, ttk.Radiobutton):
                widget.configure(state=state)

    def setup_keyboard_shortcuts(self):
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())

        # State shortcuts for Fighter 1
        self.root.bind("1", lambda e: self.set_state(1, FrameState.NEUTRAL))
        self.root.bind("2", lambda e: self.set_state(1, FrameState.STARTUP))
        self.root.bind("3", lambda e: self.set_state(1, FrameState.ACTIVE))
        self.root.bind("4", lambda e: self.set_state(1, FrameState.RECOVERY))
        self.root.bind("5", lambda e: self.set_state(1, FrameState.HITSTUN))
        self.root.bind("6", lambda e: self.set_state(1, FrameState.BLOCKSTUN))
        self.root.bind("7", lambda e: self.set_state(1, FrameState.PARRY))
        self.root.bind("8", lambda e: self.set_state(1, FrameState.DODGE))

        # State shortcuts for Fighter 2
        self.root.bind("q", lambda e: self.set_state(2, FrameState.NEUTRAL))
        self.root.bind("w", lambda e: self.set_state(2, FrameState.STARTUP))
        self.root.bind("e", lambda e: self.set_state(2, FrameState.ACTIVE))
        self.root.bind("r", lambda e: self.set_state(2, FrameState.RECOVERY))
        self.root.bind("t", lambda e: self.set_state(2, FrameState.HITSTUN))
        self.root.bind("y", lambda e: self.set_state(2, FrameState.BLOCKSTUN))
        self.root.bind("u", lambda e: self.set_state(2, FrameState.PARRY))
        self.root.bind("i", lambda e: self.set_state(2, FrameState.DODGE))

    def open_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize complete frame history with neutral states
            self.fighter1_states = [FrameState.NEUTRAL] * self.total_frames
            self.fighter2_states = [FrameState.NEUTRAL] * self.total_frames

            # Reset the meter buffers
            self.meter_fighter1_states = [FrameState.NEUTRAL] * self.meter_size
            self.meter_fighter2_states = [FrameState.NEUTRAL] * self.meter_size
            self.meter_position = 0

            # Reset window start for write mode sliding window
            self.window_start = 0

            # Reset to first frame
            self.current_frame = 0

            self.frame_slider.configure(to=self.total_frames - 1)
            self.update_frame()
            self.status_var.set(f"Loaded video: {os.path.basename(file_path)}")

    def update_frame(self):
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.write_mode:
            # For write mode - update the sliding window
            # Adjust window start to keep current frame in view
            if self.current_frame >= self.meter_size:
                self.window_start = max(0, self.current_frame - self.meter_size + 1)
            else:
                self.window_start = 0

            # Use chronological window for display
            for i in range(self.meter_size):
                frame_idx = self.window_start + i
                if frame_idx < self.total_frames:
                    self.meter_fighter1_states[i] = self.fighter1_states[frame_idx]
                    self.meter_fighter2_states[i] = self.fighter2_states[frame_idx]
                else:
                    self.meter_fighter1_states[i] = FrameState.NEUTRAL
                    self.meter_fighter2_states[i] = FrameState.NEUTRAL

            # Set pointer position relative to window start
            self.meter_position = self.current_frame - self.window_start
        else:
            # For read mode - use circular buffer
            self.meter_position = self.current_frame % self.meter_size
            self.meter_fighter1_states[self.meter_position] = self.fighter1_states[
                self.current_frame
            ]
            self.meter_fighter2_states[self.meter_position] = self.fighter2_states[
                self.current_frame
            ]

        # Update radio button states to reflect current frame data only in read mode
        if not self.write_mode:
            self.f1_state.set(self.fighter1_states[self.current_frame])
            self.f2_state.set(self.fighter2_states[self.current_frame])

        # Add frame meter overlay
        frame = self.add_frame_meter(frame)

        # Update frame counter
        self.frame_label.configure(
            text=f"Frame: {self.current_frame}/{self.total_frames-1}"
        )

        # Display frame
        height, width = frame.shape[:2]
        img = Image.fromarray(frame)
        self.current_img_tk = ImageTk.PhotoImage(image=img)

        # Resize canvas to match frame size while maintaining aspect ratio
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            scale = min(canvas_width / width, canvas_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_img_tk = ImageTk.PhotoImage(image=img)

            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2

            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                x, y, image=self.current_img_tk, anchor=tk.NW
            )

    def add_frame_meter(self, frame):
        height, width = frame.shape[:2]
        meter_height = 20
        gap_height = 5  # Gap between P1 and P2 meters
        meter_y = height - (meter_height * 2 + gap_height) - 10  # Move up by 10px

        # Create overlay
        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        # Calculate the width of each state block
        state_width = width // self.meter_size

        # Draw fighter 1 meter (top) - using the meter buffer
        for i in range(self.meter_size):
            # Get the state from our meter buffer
            state = self.meter_fighter1_states[i]
            color = STATE_COLORS.get(state, (255, 255, 255))

            x1 = i * state_width
            x2 = (i + 1) * state_width

            # Draw filled rectangle
            cv2.rectangle(
                overlay, (x1, meter_y), (x2, meter_y + meter_height), (*color, 230), -1
            )
            # Draw black border
            cv2.rectangle(
                overlay, (x1, meter_y), (x2, meter_y + meter_height), (0, 0, 0, 255), 1
            )

        # Draw fighter 2 meter (bottom) - using the meter buffer
        for i in range(self.meter_size):
            # Get the state from our meter buffer
            state = self.meter_fighter2_states[i]
            color = STATE_COLORS.get(state, (255, 255, 255))

            x1 = i * state_width
            x2 = (i + 1) * state_width

            # Draw filled rectangle
            cv2.rectangle(
                overlay,
                (x1, meter_y + meter_height + gap_height),
                (x2, meter_y + meter_height * 2 + gap_height),
                (*color, 230),
                -1,
            )
            # Draw black border
            cv2.rectangle(
                overlay,
                (x1, meter_y + meter_height + gap_height),
                (x2, meter_y + meter_height * 2 + gap_height),
                (0, 0, 0, 255),
                1,
            )

        # Add text for fighter labels
        cv2.putText(
            overlay,
            f"P1",
            (2, meter_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0, 255),
            1,
        )
        cv2.putText(
            overlay,
            f"P2",
            (2, meter_y + meter_height + gap_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0, 255),
            1,
        )

        # Draw indicator for current frame position
        x_pos = self.meter_position * state_width + state_width // 2

        # Draw triangle indicator above P1 meter
        triangle_pts = np.array(
            [
                [x_pos, meter_y - 5],
                [x_pos - 5, meter_y - 10],
                [x_pos + 5, meter_y - 10],
            ],
            np.int32,
        )
        cv2.fillPoly(overlay, [triangle_pts], (255, 255, 255, 255))

        # Blend overlay with frame
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):
            frame_rgba[:, :, c] = (
                frame_rgba[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
            )

        return frame_rgba

    def update_fighter_state(self, fighter_num):
        if not self.write_mode:
            # In read mode, just update the UI to show the current state
            if fighter_num == 1:
                self.f1_state.set(self.fighter1_states[self.current_frame])
            else:
                self.f2_state.set(self.fighter2_states[self.current_frame])
            return

        # In write mode, proceed with normal updates
        state = self.f1_state.get() if fighter_num == 1 else self.f2_state.get()

        # Update frame data in the complete history
        if fighter_num == 1:
            self.fighter1_states[self.current_frame] = state
        else:
            self.fighter2_states[self.current_frame] = state

        # Update the frame display without advancing the frame
        self.update_frame()

    def set_state(self, fighter_num, state):
        """Set the state for a fighter without advancing the frame."""
        if fighter_num == 1:
            self.f1_state.set(state)
        else:
            self.f2_state.set(state)
        self.update_fighter_state(fighter_num)

    def save_data(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        try:
            # Save just the fighter states
            data = {
                "fighter1_states": self.fighter1_states,
                "fighter2_states": self.fighter2_states,
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            self.status_var.set(f"Saved frame data to {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save frame data: {str(e)}")
            self.status_var.set("Error saving frame data")

    def load_data(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return

        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Load the complete state arrays
                self.fighter1_states = data["fighter1_states"]
                self.fighter2_states = data["fighter2_states"]

                # Update frame display
                self.update_frame()

                self.status_var.set(f"Loaded frames from {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load frames: {str(e)}")
                self.status_var.set("Error loading frame data")

    def export_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")]
        )

        if file_path:
            # Create output video writer
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

            # Process each frame
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Save current state
            current_frame = self.current_frame
            current_write_mode = self.write_mode
            current_window_start = self.window_start

            # Reset to beginning
            self.current_frame = 0
            self.window_start = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Update the frame display for the current frame
                if self.write_mode:
                    # For write mode - update the sliding window
                    if self.current_frame >= self.meter_size:
                        self.window_start = max(
                            0, self.current_frame - self.meter_size + 1
                        )
                    else:
                        self.window_start = 0

                    # Use chronological window for display
                    for i in range(self.meter_size):
                        frame_idx = self.window_start + i
                        if frame_idx < total_frames:
                            self.meter_fighter1_states[i] = self.fighter1_states[
                                frame_idx
                            ]
                            self.meter_fighter2_states[i] = self.fighter2_states[
                                frame_idx
                            ]
                        else:
                            self.meter_fighter1_states[i] = FrameState.NEUTRAL
                            self.meter_fighter2_states[i] = FrameState.NEUTRAL

                    # Set pointer position relative to window start
                    self.meter_position = self.current_frame - self.window_start
                else:
                    # For read mode - use circular buffer
                    self.meter_position = self.current_frame % self.meter_size
                    self.meter_fighter1_states[self.meter_position] = (
                        self.fighter1_states[self.current_frame]
                    )
                    self.meter_fighter2_states[self.meter_position] = (
                        self.fighter2_states[self.current_frame]
                    )

                # Add frame meter overlay
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.add_frame_meter(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                out.write(frame)

                # Update frame
                self.current_frame += 1

                # Update progress
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                self.status_var.set(f"Exporting video: {progress:.1f}%")
                self.root.update()

            # Restore original state
            self.current_frame = current_frame
            self.write_mode = current_write_mode
            self.window_start = current_window_start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            # Update the display
            self.update_frame()

            cap.release()
            out.release()

            self.status_var.set(f"Exported video to {os.path.basename(file_path)}")

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="▮▮" if self.playing else "▶")
        if self.playing:
            self.play()

    def play(self):
        if self.playing:
            if self.write_mode:
                # Update the current frame's states based on radio buttons BEFORE advancing
                current_f1_state = self.f1_state.get()
                current_f2_state = self.f2_state.get()

                # Update the current frame data with the current radio button selections
                self.fighter1_states[self.current_frame] = current_f1_state
                self.fighter2_states[self.current_frame] = current_f2_state

            # Advance to next frame
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            # Update UI
            self.frame_slider.set(self.current_frame)
            self.update_frame()

            # Schedule next frame
            self.root.after(int(1000 / self.fps), self.play)

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            if self.write_mode:
                # Update the current frame's states based on radio buttons BEFORE advancing
                current_f1_state = self.f1_state.get()
                current_f2_state = self.f2_state.get()

                # Update the current frame data with the current radio button selections
                self.fighter1_states[self.current_frame] = current_f1_state
                self.fighter2_states[self.current_frame] = current_f2_state

            # Advance to next frame
            self.current_frame += 1

            # Update UI - this will handle meter updates
            self.frame_slider.set(self.current_frame)
            self.update_frame()

    def prev_frame(self):
        if self.current_frame > 0:
            if self.write_mode:
                # Update the current frame's states based on radio buttons BEFORE moving back
                current_f1_state = self.f1_state.get()
                current_f2_state = self.f2_state.get()

                # Update the current frame data with the current radio button selections
                self.fighter1_states[self.current_frame] = current_f1_state
                self.fighter2_states[self.current_frame] = current_f2_state

            # Move back
            self.current_frame -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            # Update UI
            self.frame_slider.set(self.current_frame)

            # In write mode, custom update without sliding window if not at leftmost position
            if self.write_mode and self.current_frame >= self.window_start:
                # Only update the position within the current window - don't slide the window
                self.meter_position = self.current_frame - self.window_start

                # Update the display frame without calling update_frame (which would slide the window)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Update radio button states
                    self.f1_state.set(self.fighter1_states[self.current_frame])
                    self.f2_state.set(self.fighter2_states[self.current_frame])

                    # Add frame meter overlay
                    frame = self.add_frame_meter(frame)

                    # Update frame counter
                    self.frame_label.configure(
                        text=f"Frame: {self.current_frame}/{self.total_frames-1}"
                    )

                    # Display frame
                    height, width = frame.shape[:2]
                    img = Image.fromarray(frame)
                    self.current_img_tk = ImageTk.PhotoImage(image=img)

                    # Resize canvas to match frame size while maintaining aspect ratio
                    canvas_width = self.video_canvas.winfo_width()
                    canvas_height = self.video_canvas.winfo_height()

                    if canvas_width > 1 and canvas_height > 1:
                        scale = min(canvas_width / width, canvas_height / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)

                        img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        self.current_img_tk = ImageTk.PhotoImage(image=img)

                        x = (canvas_width - new_width) // 2
                        y = (canvas_height - new_height) // 2

                        self.video_canvas.delete("all")
                        self.video_canvas.create_image(
                            x, y, image=self.current_img_tk, anchor=tk.NW
                        )
            else:
                # Call the regular update method for other cases (read mode or when we need to slide the window)
                self.update_frame()

    def slider_changed(self, value):
        try:
            frame = int(float(value))
            if frame != self.current_frame:
                old_frame = self.current_frame

                if self.write_mode:
                    # Update the current frame's states based on radio buttons BEFORE changing frame
                    current_f1_state = self.f1_state.get()
                    current_f2_state = self.f2_state.get()

                    # Update the current frame data with the current radio button selections
                    self.fighter1_states[self.current_frame] = current_f1_state
                    self.fighter2_states[self.current_frame] = current_f2_state

                # Change to new frame
                self.current_frame = frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

                # In write mode, maintain window position if moving backwards within current window
                if self.write_mode and frame < old_frame and frame >= self.window_start:
                    # Only update the position within the current window - don't slide the window
                    self.meter_position = self.current_frame - self.window_start

                    # Update the display frame without calling update_frame (which would slide the window)
                    ret, frame_img = self.cap.read()
                    if ret:
                        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

                        # Update radio button states
                        self.f1_state.set(self.fighter1_states[self.current_frame])
                        self.f2_state.set(self.fighter2_states[self.current_frame])

                        # Add frame meter overlay
                        frame_img = self.add_frame_meter(frame_img)

                        # Update frame counter
                        self.frame_label.configure(
                            text=f"Frame: {self.current_frame}/{self.total_frames-1}"
                        )

                        # Display frame
                        height, width = frame_img.shape[:2]
                        img = Image.fromarray(frame_img)
                        self.current_img_tk = ImageTk.PhotoImage(image=img)

                        # Resize canvas to match frame size while maintaining aspect ratio
                        canvas_width = self.video_canvas.winfo_width()
                        canvas_height = self.video_canvas.winfo_height()

                        if canvas_width > 1 and canvas_height > 1:
                            scale = min(canvas_width / width, canvas_height / height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)

                            img = img.resize(
                                (new_width, new_height), Image.Resampling.LANCZOS
                            )
                            self.current_img_tk = ImageTk.PhotoImage(image=img)

                            x = (canvas_width - new_width) // 2
                            y = (canvas_height - new_height) // 2

                            self.video_canvas.delete("all")
                            self.video_canvas.create_image(
                                x, y, image=self.current_img_tk, anchor=tk.NW
                            )
                else:
                    # Call the regular update method for other cases
                    self.update_frame()
        except Exception as e:
            print(f"Error in slider change: {e}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = FrameDataAnnotator()
    app.run()
