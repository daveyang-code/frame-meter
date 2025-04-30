# Frame Data Meter Annotator

Python-based GUI tool for annotating video frame data for two fighters, often useful in analyzing frame-by-frame gameplay footage for fighting games. It provides annotation, playback, data saving/loading, and video export functionality, including an interactive "frame meter" overlay for visual reference.

# Features

- Load and play video files.

- Navigate through frames with playback controls.

- Annotate fighter states (e.g., NEUTRAL, STARTUP, ACTIVE, etc.) frame-by-frame.

- Save and load frame annotations as JSON.

- Toggle between read mode and write mode:

  - Read mode: Uses a circular buffer for real-time analysis.

  - Write mode: Uses a sliding window to annotate a fixed range of previous frames.

- Export the annotated video with an overlaid frame meter showing each fighter’s frame states.

# Dependencies
Make sure you have the following installed:
```
pip install opencv-python numpy tkinter pillow
```

# Images
![Screenshot from 2025-04-24 10-10-48](https://github.com/user-attachments/assets/e24933a5-d732-4191-814a-89e26b1109ba)
