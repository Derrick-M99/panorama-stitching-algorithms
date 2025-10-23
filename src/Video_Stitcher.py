import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to extract frames from the video at regular intervals
def extract_frames(video_path, frame_interval=30):
    start_time = time.time()
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frames.append(frame)
        success, frame = video.read()
        count += 1
    video.release()
    end_time = time.time()
    print(f"Frame extraction: {len(frames)} frames extracted in {end_time - start_time:.2f} seconds.")
    return frames

# Function to stitch images to form a panorama
def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    start_time = time.time()
    status, panorama = stitcher.stitch(images)
    end_time = time.time()
    if status == cv2.Stitcher_OK:
        print(f"Image stitching completed in {end_time - start_time:.2f} seconds.")
        return panorama
    else:
        print(f"Error during stitching: {status}")
        return None

# Display image with Matplotlib
def plot_image(image, title="Final Video Panorama", figsize=(20, 10)):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

video_path = 'Data/data_3.MP4'

# Step 1: Extract frames from the video
frames = extract_frames(video_path, frame_interval=30)

# Step 2: Stitch the extracted frames into a panorama
if len(frames) > 1:
    panorama = stitch_images(frames)
    if panorama is not None:
        plot_image(panorama, "Final Stitched Panorama")
    else:
        print("Panorama stitching failed.")
else:
    print("Not enough frames extracted from the video for panorama creation.")
