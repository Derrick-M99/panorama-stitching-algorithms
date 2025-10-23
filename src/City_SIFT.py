import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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

def detect_features(image):
    """
    Detect SIFT features and compute descriptors.
    """
    start_time = time.time()
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    end_time = time.time()
    print(f"Feature detection: {len(keypoints)} keypoints detected in {end_time - start_time:.2f} seconds.")
    return keypoints, descriptors

def match_features(des1, des2):
    """
    Match SIFT descriptors using FLANN-based matcher.
    """
    start_time = time.time()
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    end_time = time.time()
    print(f"Feature matching: {len(good_matches)} matches found in {end_time - start_time:.2f} seconds.")
    return good_matches

def compute_homography(matches, kp1, kp2):
    """
    Compute homography matrix using RANSAC.
    """
    if len(matches) < 10:  
        print("Not enough matches to compute homography.")
        return None
    start_time = time.time()
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    end_time = time.time()
    print(f"Homography computation: Completed in {end_time - start_time:.2f} seconds.")
    return H

def stitch_images(images):
    """
    Stitches a list of images into a panorama.
    """
    start_time = time.time()
    base_img = images[0]
    h_base, w_base = base_img.shape[:2]
    base_img_corners = np.float32([[0, 0], [w_base, 0], [w_base, h_base], [0, h_base]]).reshape(-1, 1, 2)

    # Compute cumulative homographies and determine the canvas size
    homographies = [np.eye(3)]
    all_corners = [base_img_corners]

    for i in range(1, len(images)):
        print(f"Processing image {i + 1}...")
        kp1, des1 = detect_features(images[i - 1])
        kp2, des2 = detect_features(images[i])
        matches = match_features(des1, des2)

        H = compute_homography(matches, kp1, kp2)
        if H is None:
            print(f"Skipping image {i} due to insufficient matches.")
            continue

        H_cumulative = np.dot(homographies[-1], H)
        homographies.append(H_cumulative)

        h, w = images[i].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H_cumulative)
        all_corners.append(transformed_corners)

    # Determine canvas size
    all_corners = np.concatenate(all_corners, axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Warp each image into the canvas
    for i, img in enumerate(images):
        print(f"Warping image {i + 1}...")
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        warped_img = cv2.warpPerspective(img, np.dot(H_translation, homographies[i]), (canvas_width, canvas_height))

        mask = (warped_img > 0).astype(np.uint8)
        canvas = np.where(mask, warped_img, canvas)

    end_time = time.time()
    print(f"Image stitching completed in {end_time - start_time:.2f} seconds.")
    return canvas

# Main Function
if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('IMG_7984.jpg')
    img2 = cv2.imread('IMG_7985.jpg')
    img3 = cv2.imread('IMG_7986.jpg')
    img1 = cv2.resize(img1, (dim_x, dim_y))
    img2 = cv2.resize(img2, (dim_x, dim_y))
    img3 = cv2.resize(img3, (dim_x, dim_y))
    images = [img1, img2, img3]

    # Visualize keypoints
    keypoints_drawn_1 = cv2.drawKeypoints(img1, detect_features(img1)[0], None, color=(0, 0, 255))
    keypoints_drawn_2 = cv2.drawKeypoints(img2, detect_features(img2)[0], None, color=(0, 0, 255))
    keypoints_drawn_3 = cv2.drawKeypoints(img3, detect_features(img3)[0], None, color=(0, 0, 255))
    keypoints_combined = np.hstack((keypoints_drawn_1, keypoints_drawn_2, keypoints_drawn_3))
    plot_image(keypoints_combined, "Keypoints")

    # Visualize matches
    matches_1_2 = match_features(detect_features(img1)[1], detect_features(img2)[1])
    matches_2_3 = match_features(detect_features(img2)[1], detect_features(img3)[1])

    matches_drawn_1_2 = cv2.drawMatches(img1, detect_features(img1)[0], img2, detect_features(img2)[0], matches_1_2[:50], None,
                                        matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matches_drawn_2_3 = cv2.drawMatches(img2, detect_features(img2)[0], img3, detect_features(img3)[0], matches_2_3[:50], None,
                                        matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plot_image(matches_drawn_1_2, "Matched Keypoints between Image 1 and Image 2")
    plot_image(matches_drawn_2_3, "Matched Keypoints between Image 2 and Image 3")

    # Stitch images into a panorama
    stitched_panorama = stitch_images(images)

    # Display the final stitched panorama
    plot_image(stitched_panorama, "Final Stitched Image")
