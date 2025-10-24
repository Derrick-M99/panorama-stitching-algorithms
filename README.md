# Panorama Stitching: A Computer Vision Project

This repository contains a comparative study of image stitching algorithms designed to create panoramas from various sources, including cityscapes, aerial drone footage, and video streams. The project implements and evaluates feature-based methods like **SIFT** and **ORB** and benchmarks them against OpenCV's built-in stitcher.

This was developed as the final project for my MSc in Robotics, AI, and Autonomous Systems.

## Key Features

* **Manual Implementation**: A from-scratch pipeline covering feature detection, matching, homography estimation (with RANSAC), and image warping.
* **Algorithm Comparison**: A direct comparison between SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF) feature detectors.
* **Performance Benchmarking**: Evaluation of the manual implementations against OpenCV's high-level `cv2.Stitcher` class.
* **Versatile Data Handling**: Scripts tailored for different scenarios, including multi-image cityscapes, two-image drone shots, and video-to-panorama conversion.


## Tech Stack

* **Language**: Python
* **Libraries**: OpenCV, NumPy, Matplotlib

## Methodology

The manual image stitching pipeline follows these four core computer vision steps:

1.  **Feature Detection**: Identify salient keypoints in each image using algorithms like SIFT (`cv2.SIFT_create()`) or ORB (`cv2.ORB_create()`).
2.  **Feature Matching**: Find corresponding keypoints between images using a Brute-Force matcher with filtering techniques like Lowe's ratio test.
3.  **Homography Estimation**: Compute the perspective transformation matrix ($H$) that maps points from one image plane to another, using `cv2.findHomography` with the RANSAC algorithm for robustness.
4.  **Image Warping & Blending**: Use the computed homography matrix ($H$) to transform the images into a common panoramic canvas via `cv2.warpPerspective`.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/panorama-stitching-algorithms.git](https://github.com/YOUR_USERNAME/panorama-stitching-algorithms.git)
    cd panorama-stitching-algorithms
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r Requirements.txt
    ```

## How to Run

Place your input images and videos in the `Data/` directory. The output images will be displayed on-screen and can be saved to the `Results/` directory.

#### Cityscape Image Stitching

* **Manual SIFT Implementation**:
    ```bash
    python City_SIFT.py
    ```
* **Manual ORB Implementation**:
    ```bash
    python City_ORB.py
    ```
* **Using OpenCV's Stitcher**:
    ```bash
    python City_CV_Stitcher.py
    ```

#### Drone Image Stitching

* **Manual SIFT Implementation**:
    ```bash
    python Drone_SIFT.py
    ```
* **Manual ORB Implementation**:
    ```bash
    python Drone_ORB.py
    ```
* **Using OpenCV's Stitcher**:
    ```bash
    python Drone_CV_Stitcher.py
    ```

#### Video to Panorama

* **Using OpenCV's Stitcher**:
    ```bash
    python Video_Stitcher.py
    ```
* **Manual Frame Extraction & Stitching**:
    ```bash
    python Manual_Video_Panorama.py
    ```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
