import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class Image_Stitching:
    def __init__(self):
        self.ratio = 0.85  
        self.min_match = 10  
        # Ensure SIFT is available
        if hasattr(cv2, 'SIFT_create'):
            self.sift = cv2.SIFT_create()
        elif hasattr(cv2, 'xfeatures2d'):
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            raise AttributeError("SIFT is not available in your OpenCV installation.")
        self.smoothing_window_size = 800

    def plot_image(self, image, title, save_path=None, figsize=(12, 6)):
        """Unified image plotting function with dtype handling."""
        # Ensure image is uint8 for OpenCV functions
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)

        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


    def visualize_keypoints(self, img1, kp1, img2, kp2):
        """Visualizes keypoints of two images side by side."""
        img1_with_kp = cv2.drawKeypoints(img1, kp1, None, color=(255, 0, 0))
        img2_with_kp = cv2.drawKeypoints(img2, kp2, None, color=(255, 0, 0))
        combined = np.hstack((img1_with_kp, img2_with_kp))
        self.plot_image(combined, "Keypoints on Images", save_path="keypoints_combined.png")

    def visualize_matches(self, img1, kp1, img2, kp2, matches):
        """Visualizes matching keypoints between two images."""
        matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.plot_image(matches_img, "Matching Keypoints", save_path="matched_keypoints.png")

    def registration(self, img1, img2):
        # Detect features in the first image
        start_time = time.time()
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        detection_time_1 = time.time() - start_time
        print(f"Feature detection: {len(kp1)} keypoints detected in {detection_time_1:.2f} seconds.")

        # Detect features in the second image
        start_time = time.time()
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        detection_time_2 = time.time() - start_time
        print(f"Feature detection: {len(kp2)} keypoints detected in {detection_time_2:.2f} seconds.")

        # Visualize keypoints
        self.visualize_keypoints(img1, kp1, img2, kp2)

        # Match features
        start_time = time.time()
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        matching_time = time.time() - start_time

        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append(m1)

        print(f"Feature matching: {len(good_matches)} matches found in {matching_time:.2f} seconds.")

        # Visualize matching keypoints
        self.visualize_matches(img1, kp1, img2, kp2, good_matches)
        
        # Compute homography
        if len(good_points) > self.min_match:
            start_time = time.time()
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            homography_time = time.time() - start_time
            print(f"Homography computation: Completed in {homography_time:.2f} seconds.")
            return H
        else:
            raise ValueError("Not enough matches found to calculate homography.")

    def create_mask(self, img1, img2, version):
        # Dimensions of the images
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        # Define smoothing parameters
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - offset
        mask = np.zeros((height_panorama, width_panorama))

        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset), (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:  # Right image
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset), (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        # Compute homography matrix
        start_time = time.time()
        H = self.registration(img1, img2)
        
        # Dimensions of the panorama
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        # Left image panorama
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1

        # Right image panorama
        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2

        # Combine the two panoramas
        result = panorama1 + panorama2

        # Crop black edges
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]

        stitching_time = time.time() - start_time
        print(f"Total stitching time: {stitching_time:.2f} seconds.")

        # Show and save final panorama
        self.plot_image(final_result, "Final Stitched Image")

        return final_result

def main(argv1=None, argv2=None):
    image_paths_2 = ['IMG_7898.jpg', 'IMG_7899.jpg']

    # Read the images
    img1 = cv2.imread(image_paths_2[0])
    img2 = cv2.imread(image_paths_2[1])

    # Check if the images were loaded successfully
    if img1 is None:
        print(f"Error: Could not load image {image_paths_2[0]}")
        return
    if img2 is None:
        print(f"Error: Could not load image {image_paths_2[1]}")
        return

    # Stitch the images
    try:
        final = Image_Stitching().blending(img1, img2)
    except Exception as e:
        print(f"Error during stitching: {e}")

if __name__ == '__main__':
    main()
