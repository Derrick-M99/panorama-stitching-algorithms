import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Define dimensions for resizing
dim = (1024, 768)

# Utility function to display the final image using matplotlib
def plot_image(image, title, figsize=(12, 6)):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Start total timer ---
total_start_time = time.time()

# Load and resize images
load_start_time = time.time()
img1 = cv2.imread('IMG_7898.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('IMG_7899.jpg', cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
load_end_time = time.time()
print(f"Image loading and resizing: Completed in {load_end_time - load_start_time:.2f} seconds.")

# Perform stitching
images = [img1, img2]
stitch_start_time = time.time()
stitcher = cv2.Stitcher.create()
ret, pano = stitcher.stitch(images)
stitch_end_time = time.time()

# Check if stitching was successful
if ret == cv2.Stitcher_OK:
    print(f"Image stitching: Completed in {stitch_end_time - stitch_start_time:.2f} seconds.")
    plot_image(pano, "Final Stitched Image")
else:
    print("Error during stitching")

# Total computation time
total_end_time = time.time()
print(f"Total computation time: {total_end_time - total_start_time:.2f} seconds.")