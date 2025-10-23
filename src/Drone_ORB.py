import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set final stitched image dimensions
dim_x, dim_y = 1024, 768

# Function to plot a single image with a title
def plot_image(image, title, figsize=(10, 5)):
    '''Display a single image with a title.'''
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title, fontsize=14)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Load images and resize to (1024, 768)
left = cv2.imread('IMG_7898.jpg')
right = cv2.imread('IMG_7899.jpg')
left = cv2.resize(left, (dim_x, dim_y))
right = cv2.resize(right, (dim_x, dim_y))

# Feature detection using ORB
orb = cv2.ORB_create(nfeatures=5000)  
kp_left, des_left = orb.detectAndCompute(left, None)
kp_right, des_right = orb.detectAndCompute(right, None)

# Visualize keypoints on each image and combine
keypoints_drawn_left = cv2.drawKeypoints(left, kp_left, None, color=(0, 0, 255))
keypoints_drawn_right = cv2.drawKeypoints(right, kp_right, None, color=(0, 0, 255))
keypoints_combined = np.hstack((keypoints_drawn_left, keypoints_drawn_right))
plot_image(keypoints_combined, "Left and Right Images with Keypoints")

# Matching features using Brute Force with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_left, des_right)

# Sort matches by distance and keep the top matches
limit = 50 
matches = sorted(matches, key=lambda x: x.distance)[:limit]

# Visualize matched keypoints 
matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, matches, None, 
                                matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plot_image(matches_drawn, "Matched Keypoints between Left and Right Images")

# Extract matched keypoints coordinates
left_pts = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
right_pts = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute the homography matrix
M, mask = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, 5.0)

# If homography could not be computed, output an error
if M is None:
    print("Error: Homography could not be computed.")
else:
    # Warp the right image using the homography matrix
    warped = cv2.warpPerspective(right, M, (dim_x * 2, dim_y))
    
    # Combine the left and warped right images
    comb = warped.copy()
    comb[0:left.shape[0], 0:left.shape[1]] = left

    # Display the final stitched image
    plot_image(comb, "Final Stitched Image")

    
