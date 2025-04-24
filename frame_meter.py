import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
from dataclasses import dataclass
import os
from ultralytics import YOLO


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

        # Player positions and detection data
        self.fighter1_position = None  # Will store player 1's detected position
        self.fighter2_position = None  # Will store player 2's detected position
        self.players_swapped = False  # Flag to indicate if players are swapped

        # Initialize YOLO model
        self.yolo_net = None
        self.yolo_classes = []
        self.initialize_yolo()

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

        # Toggle for detection visualization
        self.show_detections = tk.BooleanVar(value=True)

        # Mask dilate kernel size
        self.mask_dilate_size = 7

        # UI elements
        self.setup_ui()

        # Keyboard shortcuts
        self.setup_keyboard_shortcuts()

    def initialize_yolo(self):
        try:
            from ultralytics import YOLO

            # Load the YOLOv8 model
            self.yolo_model = YOLO("yolov8n.pt")

            # Set model to evaluation mode
            self.yolo_model.fuse()

            # COCO class names (person is class 0)
            self.yolo_classes = ["person"]

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

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

        # Add toggle for detections
        det_toggle = ttk.Checkbutton(
            toolbar,
            text="Show Detections",
            variable=self.show_detections,
            command=self.update_frame,
        )
        det_toggle.pack(side=tk.RIGHT, padx=5)

        swap_button = ttk.Button(
            toolbar, text="Swap Players", command=self.swap_players
        )
        swap_button.pack(side=tk.RIGHT, padx=5)

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

    def swap_players(self):
        self.players_swapped = not self.players_swapped
        self.status_var.set(
            f"Players {'swapped' if self.players_swapped else 'reset to default'}"
        )
        self.update_frame()

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
        self.root.bind("m", lambda e: self.toggle_detections())

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

        self.root.bind("s", lambda e: self.swap_players())

    def toggle_detections(self):
        self.show_detections.set(not self.show_detections.get())
        self.update_frame()

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

    def detect_players(self, frame):
        height, width = frame.shape[:2]

        # Create empty masks for both fighters
        fighter1_mask = np.zeros((height, width), dtype=np.uint8)
        fighter2_mask = np.zeros((height, width), dtype=np.uint8)

        # Try YOLO detection first
        if self.yolo_model is not None:
            # Run inference
            results = self.yolo_model(frame, verbose=False)

            # Process results
            boxes = []
            for result in results:
                for box in result.boxes:
                    # Only consider 'person' class (id 0)
                    if box.cls == 0 and box.conf > 0.5:
                        # Get box coordinates in (x1, y1, x2, y2) format
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append(
                            [x1, y1, x2 - x1, y2 - y1]
                        )  # Convert to (x, y, w, h)

            # Sort boxes by x-coordinate (left to right)
            boxes.sort(key=lambda b: b[0])

            # We only need the first two detections (leftmost and rightmost)
            if len(boxes) > 2:
                boxes = [boxes[0], boxes[-1]]

            # Create masks for each fighter
            for i, box in enumerate(boxes):
                x, y, w, h = box
                if i == 0:  # Leftmost box (fighter 1)
                    cv2.rectangle(fighter1_mask, (x, y), (x + w, y + h), 255, -1)
                else:  # Rightmost box (fighter 2)
                    cv2.rectangle(fighter2_mask, (x, y), (x + w, y + h), 255, -1)

        # If YOLO didn't find anything or isn't available, use simple bounding boxes
        if np.max(fighter1_mask) == 0 and np.max(fighter2_mask) == 0:
            # Simple fallback: divide screen into left and right halves
            left_side_mask = np.zeros((height, width), dtype=np.uint8)
            left_side_mask[:, : width // 2] = 255
            fighter1_mask = left_side_mask

            right_side_mask = np.zeros((height, width), dtype=np.uint8)
            right_side_mask[:, width // 2 :] = 255
            fighter2_mask = right_side_mask

        # Apply dilation to the masks
        kernel = np.ones((self.mask_dilate_size, self.mask_dilate_size), np.uint8)
        fighter1_mask = cv2.dilate(fighter1_mask, kernel, iterations=1)
        fighter2_mask = cv2.dilate(fighter2_mask, kernel, iterations=1)

        # Make sure masks don't overlap
        overlap = cv2.bitwise_and(fighter1_mask, fighter2_mask)
        fighter1_mask = cv2.bitwise_and(fighter1_mask, cv2.bitwise_not(overlap))

        # Swap masks if players are swapped
        if self.players_swapped:
            fighter1_mask, fighter2_mask = fighter2_mask, fighter1_mask

        return fighter1_mask, fighter2_mask

    def apply_player_masks(self, frame, fighter1_mask, fighter2_mask):
        if not self.show_detections.get():
            return frame

        # Get current states
        f1_state = self.fighter1_states[self.current_frame]
        f2_state = self.fighter2_states[self.current_frame]

        # Get colors for states (BGR for OpenCV)
        f1_color = STATE_COLORS.get(f1_state, (26, 26, 26))
        f2_color = STATE_COLORS.get(f2_state, (26, 26, 26))

        # Convert to BGR (OpenCV uses BGR)
        f1_color_bgr = (f1_color[2], f1_color[1], f1_color[0])
        f2_color_bgr = (f2_color[2], f2_color[1], f2_color[0])

        # Create colored masks
        height, width = frame.shape[:2]
        f1_colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        f2_colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply colors to masks
        f1_colored_mask[fighter1_mask > 0] = f1_color_bgr
        f2_colored_mask[fighter2_mask > 0] = f2_color_bgr

        # Create alpha channel (50% transparent)
        alpha = 0.5

        # Apply the masks to the frame
        mask_overlay = cv2.addWeighted(
            f1_colored_mask, alpha, f2_colored_mask, alpha, 0
        )
        result = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)

        return result

    def update_frame(self):
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

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

        # Detect players and create masks
        fighter1_mask, fighter2_mask = self.detect_players(frame)

        # Apply colored masks based on frame states
        frame = self.apply_player_masks(frame, fighter1_mask, fighter2_mask)

        # Convert frame to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        gap_height = 5
        meter_y = height - (meter_height * 2 + gap_height) - 10  # 10px from bottom

        # Create overlay
        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        # Calculate meter dimensions
        meter_width = int(width * 0.9)
        # Adjust width to be exactly divisible by meter_size for perfect blocks
        meter_width = (meter_width // self.meter_size) * self.meter_size

        # Calculate starting position for perfect centering
        meter_x_start = (width - meter_width) // 2
        # Ensure we don't go negative (for very small frames)
        meter_x_start = max(0, meter_x_start)

        # Calculate exact block width (now perfectly divisible)
        state_width = meter_width // self.meter_size

        # Draw both meters
        for fighter_num in [1, 2]:
            y_start = (
                meter_y if fighter_num == 1 else meter_y + meter_height + gap_height
            )

            for i in range(self.meter_size):
                state = (
                    self.meter_fighter1_states[i]
                    if fighter_num == 1
                    else self.meter_fighter2_states[i]
                )
                color = STATE_COLORS.get(state, (255, 255, 255))

                x1 = meter_x_start + i * state_width
                x2 = x1 + state_width

                # Draw rectangle
                cv2.rectangle(
                    overlay,
                    (x1, y_start),
                    (x2, y_start + meter_height),
                    (*color, 230),
                    -1,
                )
                cv2.rectangle(
                    overlay,
                    (x1, y_start),
                    (x2, y_start + meter_height),
                    (0, 0, 0, 255),
                    1,
                )

        # Current position indicator (centered on active block)
        active_block_center = (
            meter_x_start + self.meter_position * state_width + state_width // 2
        )
        cv2.fillPoly(
            overlay,
            [
                np.array(
                    [
                        [active_block_center, meter_y - 5],
                        [active_block_center - 5, meter_y - 10],
                        [active_block_center + 5, meter_y - 10],
                    ],
                    np.int32,
                )
            ],
            (255, 255, 255, 255),
        )

        # Blend overlay
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        alpha = overlay[..., 3] / 255.0
        for c in range(3):
            frame_rgba[..., c] = (
                frame_rgba[..., c] * (1 - alpha) + overlay[..., c] * alpha
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
            current_show_detections = self.show_detections.get()

            # Reset to beginning
            self.current_frame = 0
            self.window_start = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_bgr = frame.copy()

                # Only apply masks if show_detections is enabled
                if current_show_detections:
                    fighter1_mask, fighter2_mask = self.detect_players(frame_bgr)
                    frame_bgr = self.apply_player_masks(
                        frame_bgr, fighter1_mask, fighter2_mask
                    )

                # Convert to RGB for frame meter overlay
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = self.add_frame_meter(frame_rgb)

                # Convert back to BGR for video writing
                if frame_rgb.shape[2] == 4:  # If RGBA
                    frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2BGR)
                else:  # If RGB
                    frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                out.write(frame_out)

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
