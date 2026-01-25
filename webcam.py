import cv2
import sys
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from hand_detection import HandDetector
import csv
import os

# Initialize hand detector
hand_detector = HandDetector(max_num_hands=2, min_detection_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    sys.exit()

# Get frame properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False
video_counter = 0

# Global flags
stop_flag = False
fullscreen = False
is_maximized = False
capture_flag = False
current_result = None
last_saved_landmarks = None

# Dataset setup
DATASET_PATH = os.path.join("dataset", "signs.csv")
os.makedirs("dataset", exist_ok=True)
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")
        writer.writerow(header)

# Try to set camera properties for better focus
try:
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
except:
    pass

def update_frame(root, label):
    global recording, out, video_counter, stop_flag, fullscreen, is_maximized, capture_flag, current_result
    
    if stop_flag:
        return
    
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        root.quit()
        return
    
    # Detect hand landmarks
    frame, result = hand_detector.detect_hands(frame)
    current_result = result
    
    if capture_flag:
        cv2.imwrite("photo.jpg", frame)
        print("Photo captured")
        capture_flag = False
    
    # Write frame to video if recording
    if recording and out is not None:
        out.write(frame)
    
    # Display recording status on frame
    if recording:
        cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "Click maximize | 's' capture | 'r' record | 'f' full | 'a-c' save | 'q' quit", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Get current window size
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    
    # Check if window is maximized
    current_screen_width = root.winfo_screenwidth()
    current_screen_height = root.winfo_screenheight()
    
    if window_width >= current_screen_width - 20 and window_height >= current_screen_height - 40:
        if not is_maximized and not fullscreen:
            is_maximized = True
            fullscreen = True
            print("Window maximized - Fullscreen mode ON")
    else:
        is_maximized = False
    
    # Resize frame to match window
    display_height = max(window_height - 40, 100) if window_height > 0 else frame.shape[0]
    display_width = max(window_width - 20, 100) if window_width > 0 else frame.shape[1]
    
    resized_frame = cv2.resize(frame, (display_width, display_height))
    
    # Convert to RGB for PIL
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Convert to PhotoImage
    photo = ImageTk.PhotoImage(image=pil_image)
    
    # Update label
    label.config(image=photo)
    label.image = photo
    
    # Schedule next frame update
    label.after(30, update_frame, root, label)

def on_key_press(event):
    global recording, out, video_counter, stop_flag, fullscreen, capture_flag, current_result, last_saved_landmarks
    
    key = event.keysym.lower()
    
    if key == 'q':
        stop_flag = True
        root.quit()
    elif key == 's':
        capture_flag = True
    elif key == 'r':
        if not recording:
            video_counter += 1
            video_name = f"video_{video_counter}.mp4"
            out = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))
            recording = True
            print(f"Recording started: {video_name}")
        else:
            recording = False
            if out is not None:
                out.release()
                out = None
            print("Recording stopped")
    elif key == 'f':
        fullscreen = not fullscreen
        if fullscreen:
            root.state('zoomed')
            print("Fullscreen ON")
        else:
            root.state('normal')
            root.geometry("640x480")
            print("Fullscreen OFF")
    elif key == 'escape':
        if fullscreen:
            fullscreen = False
            root.state('normal')
            root.geometry("640x480")
            print("Fullscreen OFF")
        else:
            stop_flag = True
            root.quit()
    elif key in ['a', 'b', 'c']:
        if current_result and current_result.multi_hand_landmarks:
            # Get the last detected hand
            hand_landmarks = current_result.multi_hand_landmarks[-1]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Check for duplicates
            if landmarks == last_saved_landmarks:
                print("Duplicate sign ignored (same frame).")
                return

            label_char = key.upper()
            with open(DATASET_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmarks + [label_char])
            last_saved_landmarks = landmarks
            print(f"Saved sign: {label_char}")

def on_close():
    global stop_flag
    stop_flag = True
    root.quit()

# Create main window
root = tk.Tk()
root.title("Webcam Window - Sign Language Recognition")
root.protocol("WM_DELETE_WINDOW", on_close)
root.geometry("640x480")

# Create label for video display
label = Label(root, bg="black")
label.pack(fill=tk.BOTH, expand=True)

# Bind keyboard events
root.bind('<Key>', on_key_press)

# Start frame update
try:
    update_frame(root, label)
    root.mainloop()
finally:
    stop_flag = True
    if recording and out is not None:
        out.release()
    hand_detector.close()
    cap.release()
    print("Application closed") 