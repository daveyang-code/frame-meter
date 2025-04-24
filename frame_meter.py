import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
from dataclasses import dataclass
from typing import Dict, Optional, List
import os
from datetime import datetime
from collections import deque

# Define the different states a fighter can be in
class FrameState:
    NEUTRAL = "neutral"
    STARTUP = "startup"
    ACTIVE = "active"
    RECOVERY = "recovery"
    HITSTUN = "hitstun"
    BLOCKSTUN = "blockstun"

# Color scheme based on SF6 (RGB format)
STATE_COLORS = {
    FrameState.NEUTRAL: (255, 255, 255),    
    FrameState.STARTUP: (67, 243, 241),      
    FrameState.ACTIVE: (194, 46, 104),         
    FrameState.RECOVERY: (3, 107, 180),       
    FrameState.HITSTUN: (246, 245, 57),      
    FrameState.BLOCKSTUN: (200, 200, 50)     
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
        
        # Frame data
        self.fighter1_frames: Dict[int, FighterFrameData] = {}
        self.fighter2_frames: Dict[int, FighterFrameData] = {}
        
        # Frame meter arrays (80 spots for each player)
        self.meter_size = 80
        self.fighter1_state_array = [FrameState.NEUTRAL] * self.meter_size
        self.fighter2_state_array = [FrameState.NEUTRAL] * self.meter_size
        self.meter_index = 0  # Current position in the circular array
        
        # UI elements
        self.setup_ui()
        
        # Keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
        # History for undo/redo
        self.history: List[tuple] = []
        self.history_index = -1
        
    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top toolbar
        toolbar = ttk.Frame(main_container)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Save Data", command=self.save_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Export Video", command=self.export_video).pack(side=tk.LEFT, padx=5)
        
        # Video display area
        self.video_frame = ttk.Frame(main_container)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(self.video_frame, bg='black')
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Playback controls
        controls = ttk.Frame(main_container)
        controls.pack(fill=tk.X, pady=10)
        
        ttk.Button(controls, text="⏮", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        self.play_button = ttk.Button(controls, text="▶", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="⏭", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        
        self.frame_slider = ttk.Scale(
            controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.slider_changed
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.frame_label = ttk.Label(controls, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # Annotation controls
        annotation_frame = ttk.LabelFrame(main_container, text="Frame Data Annotation")
        annotation_frame.pack(fill=tk.X, pady=10)
        
        # Fighter 1 controls
        f1_frame = ttk.Frame(annotation_frame)
        f1_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(f1_frame, text="Fighter 1:").pack(side=tk.LEFT)
        self.f1_state = tk.StringVar(value=FrameState.NEUTRAL)
        for state in [FrameState.NEUTRAL, FrameState.STARTUP, FrameState.ACTIVE,
                     FrameState.RECOVERY, FrameState.HITSTUN, FrameState.BLOCKSTUN]:
            ttk.Radiobutton(
                f1_frame,
                text=state,
                value=state,
                variable=self.f1_state,
                command=lambda: self.update_fighter_state(1)
            ).pack(side=tk.LEFT, padx=5)
        
        # Fighter 2 controls
        f2_frame = ttk.Frame(annotation_frame)
        f2_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(f2_frame, text="Fighter 2:").pack(side=tk.LEFT)
        self.f2_state = tk.StringVar(value=FrameState.NEUTRAL)
        for state in [FrameState.NEUTRAL, FrameState.STARTUP, FrameState.ACTIVE,
                     FrameState.RECOVERY, FrameState.HITSTUN, FrameState.BLOCKSTUN]:
            ttk.Radiobutton(
                f2_frame,
                text=state,
                value=state,
                variable=self.f2_state,
                command=lambda: self.update_fighter_state(2)
            ).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_container, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
    def setup_keyboard_shortcuts(self):
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        
        # State shortcuts for Fighter 1
        self.root.bind('1', lambda e: self.set_state(1, FrameState.NEUTRAL))
        self.root.bind('2', lambda e: self.set_state(1, FrameState.STARTUP))
        self.root.bind('3', lambda e: self.set_state(1, FrameState.ACTIVE))
        self.root.bind('4', lambda e: self.set_state(1, FrameState.RECOVERY))
        self.root.bind('5', lambda e: self.set_state(1, FrameState.HITSTUN))
        self.root.bind('6', lambda e: self.set_state(1, FrameState.BLOCKSTUN))
        
        # State shortcuts for Fighter 2
        self.root.bind('q', lambda e: self.set_state(2, FrameState.NEUTRAL))
        self.root.bind('w', lambda e: self.set_state(2, FrameState.STARTUP))
        self.root.bind('e', lambda e: self.set_state(2, FrameState.ACTIVE))
        self.root.bind('r', lambda e: self.set_state(2, FrameState.RECOVERY))
        self.root.bind('t', lambda e: self.set_state(2, FrameState.HITSTUN))
        self.root.bind('y', lambda e: self.set_state(2, FrameState.BLOCKSTUN))
        
        # Undo/Redo
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        
    def open_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.frame_slider.configure(to=self.total_frames - 1)
            self.update_frame()
            self.status_var.set(f"Loaded video: {os.path.basename(file_path)}")
            
    def update_frame(self):
        if self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add frame meter overlay
        frame = self.add_frame_meter(frame)
        
        # Update frame counter
        self.frame_label.configure(text=f"Frame: {self.current_frame}/{self.total_frames-1}")
        
        # Display frame
        height, width = frame.shape[:2]
        img = Image.fromarray(frame)
        self.current_img_tk = ImageTk.PhotoImage(image=img)
        
        # Resize canvas to match frame size while maintaining aspect ratio
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale = min(canvas_width/width, canvas_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_img_tk = ImageTk.PhotoImage(image=img)
            
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(x, y, image=self.current_img_tk, anchor=tk.NW)
            
    def add_frame_meter(self, frame):
        height, width = frame.shape[:2]
        meter_height = 20
        gap_height = 5  # Gap between P1 and P2 meters
        meter_y = height - (meter_height * 2 + gap_height) - 10  # Move up by 10px
        
        # Create overlay
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Get current states
        f1_state = self.fighter1_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
        f2_state = self.fighter2_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
        
        # Calculate the width of each state block
        state_width = width // 80
        
        # Draw fighter 1 meter (top)
        for i, state in enumerate(self.fighter1_state_array):
            color = STATE_COLORS.get(state, (255, 255, 255))
            x1 = i * state_width
            x2 = (i + 1) * state_width
            # Draw filled rectangle
            cv2.rectangle(overlay, (x1, meter_y), (x2, meter_y + meter_height),
                         (*color, 230), -1)
            # Draw black border
            cv2.rectangle(overlay, (x1, meter_y), (x2, meter_y + meter_height),
                         (0, 0, 0, 255), 1)
        
        # Draw fighter 2 meter (bottom)
        for i, state in enumerate(self.fighter2_state_array):
            color = STATE_COLORS.get(state, (255, 255, 255))
            x1 = i * state_width
            x2 = (i + 1) * state_width
            # Draw filled rectangle
            cv2.rectangle(overlay, (x1, meter_y + meter_height + gap_height), (x2, meter_y + meter_height * 2 + gap_height),
                         (*color, 230), -1)
            # Draw black border
            cv2.rectangle(overlay, (x1, meter_y + meter_height + gap_height), (x2, meter_y + meter_height * 2 + gap_height),
                         (0, 0, 0, 255), 1)
        
        # Add text for current states
        cv2.putText(overlay, f"P1", (2, meter_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1)
        cv2.putText(overlay, f"P2", (2, meter_y + meter_height + gap_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1)
        
        # Blend overlay with frame
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        alpha = overlay[:, :, 3] / 255.0
        
        for c in range(3):
            frame_rgba[:, :, c] = frame_rgba[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
            
        return frame_rgba
        
    def update_fighter_state(self, fighter_num):
        state = self.f1_state.get() if fighter_num == 1 else self.f2_state.get()
        
        # Save current state to history
        self.save_to_history()
        
        # Update frame data
        if fighter_num == 1:
            self.fighter1_frames[self.current_frame] = FighterFrameData(
                frame=self.current_frame,
                state=state
            )
            # Replace the most recent entry in the array with the new state
            self.fighter1_state_array[self.meter_index] = state
        else:
            self.fighter2_frames[self.current_frame] = FighterFrameData(
                frame=self.current_frame,
                state=state
            )
            # Replace the most recent entry in the array with the new state
            self.fighter2_state_array[self.meter_index] = state
            
        # Update the frame display without advancing the frame
        self.update_frame()
        
    def set_state(self, fighter_num, state):
        """Set the state for a fighter without advancing the frame."""
        if fighter_num == 1:
            self.f1_state.set(state)
        else:
            self.f2_state.set(state)
        self.update_fighter_state(fighter_num)
        
    def save_to_history(self):
        # Save current state before making changes
        self.history = self.history[:self.history_index + 1]
        self.history.append((
            self.current_frame,
            self.fighter1_frames.copy(),
            self.fighter2_frames.copy(),
            self.fighter1_state_array.copy(),
            self.fighter2_state_array.copy()
        ))
        self.history_index = len(self.history) - 1
        
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            _, f1_frames, f2_frames, f1_array, f2_array = self.history[self.history_index]
            self.fighter1_frames = f1_frames.copy()
            self.fighter2_frames = f2_frames.copy()
            self.fighter1_state_array = f1_array.copy()
            self.fighter2_state_array = f2_array.copy()
            self.update_frame()
            
    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            _, f1_frames, f2_frames, f1_array, f2_array = self.history[self.history_index]
            self.fighter1_frames = f1_frames.copy()
            self.fighter2_frames = f2_frames.copy()
            self.fighter1_state_array = f1_array.copy()
            self.fighter2_state_array = f2_array.copy()
            self.update_frame()
            
    def save_data(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            data = {
                "video_path": self.video_path,
                "fighter1": [
                    {
                        "frame": k,
                        "state": v.state
                    }
                    for k, v in self.fighter1_frames.items()
                ],
                "fighter2": [
                    {
                        "frame": k,
                        "state": v.state
                    }
                    for k, v in self.fighter2_frames.items()
                ],
                "fighter1_state_array": self.fighter1_state_array.copy(),
                "fighter2_state_array": self.fighter2_state_array.copy()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.status_var.set(f"Saved frame data to {os.path.basename(file_path)}")
            
    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Load video if different
            if data["video_path"] != self.video_path:
                self.video_path = data["video_path"]
                self.cap = cv2.VideoCapture(self.video_path)
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_slider.configure(to=self.total_frames - 1)
                
            # Load frame data
            self.fighter1_frames = {
                entry["frame"]: FighterFrameData(
                    frame=entry["frame"],
                    state=entry["state"]
                )
                for entry in data["fighter1"]
            }
            
            self.fighter2_frames = {
                entry["frame"]: FighterFrameData(
                    frame=entry["frame"],
                    state=entry["state"]
                )
                for entry in data["fighter2"]
            }
            
            # Load state arrays if they exist
            if "fighter1_state_array" in data:
                self.fighter1_state_array = data["fighter1_state_array"].copy()
            if "fighter2_state_array" in data:
                self.fighter2_state_array = data["fighter2_state_array"].copy()
            
            self.update_frame()
            self.status_var.set(f"Loaded frame data from {os.path.basename(file_path)}")
            
    def export_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )
        
        if file_path:
            # Create output video writer
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            # Process each frame
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Add frame meter overlay
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.add_frame_meter(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                out.write(frame)
                
                # Update progress
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                self.status_var.set(f"Exporting video: {progress:.1f}%")
                self.root.update()
                
            cap.release()
            out.release()
            
            self.status_var.set(f"Exported video to {os.path.basename(file_path)}")
            
    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="⏸" if self.playing else "▶")
        if self.playing:
            self.play()
            
    def play(self):
        """Play the video."""
        if self.playing:
            # Save current states before advancing
            prev_f1_state = self.fighter1_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
            prev_f2_state = self.fighter2_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
            
            # Advance to next frame
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            # Get new states - use current radio button selection if no frame data exists
            if self.current_frame in self.fighter1_frames:
                new_f1_state = self.fighter1_frames[self.current_frame].state
            else:
                new_f1_state = self.f1_state.get()
                
            if self.current_frame in self.fighter2_frames:
                new_f2_state = self.fighter2_frames[self.current_frame].state
            else:
                new_f2_state = self.f2_state.get()
            
            # Update meter index for circular array
            self.meter_index = (self.meter_index + 1) % self.meter_size
            
            # Update state arrays
            self.fighter1_state_array[self.meter_index] = new_f1_state
            self.fighter2_state_array[self.meter_index] = new_f2_state
            
            # Update UI
            self.frame_slider.set(self.current_frame)
            self.update_frame()
            
            # Schedule next frame
            self.root.after(int(1000 / self.fps), self.play)
            
    def next_frame(self):
        """Move to the next frame."""
        if self.current_frame < self.total_frames - 1:
            # Save current states before advancing
            prev_f1_state = self.fighter1_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
            prev_f2_state = self.fighter2_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
            
            # Advance to next frame
            self.current_frame += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            # Get new states - use current radio button selection if no frame data exists
            if self.current_frame in self.fighter1_frames:
                new_f1_state = self.fighter1_frames[self.current_frame].state
            else:
                new_f1_state = self.f1_state.get()
                
            if self.current_frame in self.fighter2_frames:
                new_f2_state = self.fighter2_frames[self.current_frame].state
            else:
                new_f2_state = self.f2_state.get()
            
            # Update meter index for circular array
            self.meter_index = (self.meter_index + 1) % self.meter_size
            
            # Update state arrays
            self.fighter1_state_array[self.meter_index] = new_f1_state
            self.fighter2_state_array[self.meter_index] = new_f2_state
            
            # Update UI
            self.frame_slider.set(self.current_frame)
            self.update_frame()
            
    def prev_frame(self):
        """Move to the previous frame."""
        if self.current_frame > 0:
            # Move to previous frame
            self.current_frame -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            # Update meter index for circular array
            self.meter_index = (self.meter_index - 1) % self.meter_size
            
            # Get states for the previous frame
            if self.current_frame in self.fighter1_frames:
                f1_state = self.fighter1_frames[self.current_frame].state
            else:
                f1_state = self.f1_state.get()
                
            if self.current_frame in self.fighter2_frames:
                f2_state = self.fighter2_frames[self.current_frame].state
            else:
                f2_state = self.f2_state.get()
            
            # Update state arrays
            self.fighter1_state_array[self.meter_index] = f1_state
            self.fighter2_state_array[self.meter_index] = f2_state
            
            # Update UI
            self.frame_slider.set(self.current_frame)
            self.update_frame()
            
    def slider_changed(self, value):
        """Handle slider value change."""
        try:
            frame = int(float(value))
            if frame != self.current_frame:
                # Save current states before changing frame
                prev_f1_state = self.fighter1_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
                prev_f2_state = self.fighter2_frames.get(self.current_frame, FighterFrameData(self.current_frame, FrameState.NEUTRAL)).state
                
                # Change to new frame
                self.current_frame = frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                
                # Get new states - use current radio button selection if no frame data exists
                if self.current_frame in self.fighter1_frames:
                    new_f1_state = self.fighter1_frames[self.current_frame].state
                else:
                    new_f1_state = self.f1_state.get()
                    
                if self.current_frame in self.fighter2_frames:
                    new_f2_state = self.fighter2_frames[self.current_frame].state
                else:
                    new_f2_state = self.f2_state.get()
                
                # Update state arrays - only add new state if it's different from previous
                if new_f1_state != prev_f1_state:
                    self.fighter1_state_array[self.meter_index] = new_f1_state
                else:
                    # If state hasn't changed, add the same state again
                    self.fighter1_state_array[self.meter_index] = prev_f1_state
                    
                if new_f2_state != prev_f2_state:
                    self.fighter2_state_array[self.meter_index] = new_f2_state
                else:
                    # If state hasn't changed, add the same state again
                    self.fighter2_state_array[self.meter_index] = prev_f2_state
                
                # Update UI
                self.update_frame()
        except Exception as e:
            print(f"Error in slider change: {e}")
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FrameDataAnnotator()
    app.run() 