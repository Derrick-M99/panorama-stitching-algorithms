import cv2
import numpy as np
import os
import shutil
import time
import matplotlib.pyplot as plt

# Extract evenly spaced frames from the video
def extract_even_frames(video_path, output_folder, frame_count=8):
    start_time = time.time()
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    intervals = np.linspace(0, total_frames - 1, frame_count, dtype=int)

    # Extract frames at calculated intervals
    for idx, frame_no in enumerate(intervals):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_folder, f"frame_{idx + 1:03d}.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")

    cap.release()
    end_time = time.time()
    print(f"Extracted {len(intervals)} frames to {output_folder} in {end_time - start_time:.2f} seconds.")

# Stitch frames side by side
def stitch_frames(output_folder):
    start_time = time.time()
    frames = [cv2.imread(os.path.join(output_folder, f))
              for f in sorted(os.listdir(output_folder)) if f.endswith(".png")]
    if not frames:
        print("No frames found to stitch!")
        return None

    # Ensure all frames have the same height
    min_height = min(frame.shape[0] for frame in frames)
    resized_frames = [cv2.resize(frame, (int(frame.shape[1] * min_height / frame.shape[0]), min_height))
                      for frame in frames]

    # Stitch frames horizontally
    panorama = np.hstack(resized_frames)

    end_time = time.time()
    print(f"Manual panorama stitching completed in {end_time - start_time:.2f} seconds.")
    return panorama

# Function to show image with matplotlib
def plot_image(image, title="Final Panorama", figsize=(20, 10)):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# ---- MAIN EXECUTION ----
video_path = "Data/data_3.MP4"
output_folder = "frames"
frame_count = 8

# Step 1: Extract frames
extract_even_frames(video_path, output_folder, frame_count=frame_count)

# Step 2: Stitch frames manually
panorama = stitch_frames(output_folder)

# Step 3: Show panorama using Matplotlib
if panorama is not None:
    plot_image(panorama, "Final Manual Panorama")
