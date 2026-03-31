import cv2
import numpy as np

# Edge Detection (Canny)
def detect_edges(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def extract_hog(img):
    # Resize to correct size (width=64, height=128)
    resized = cv2.resize(img, (64, 128))

    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    features = hog.compute(resized)
    return features


# Harris Corner Detection
def detect_corners(img, original):
    corners = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
    original[corners > 0.01 * corners.max()] = [0, 0, 255]
    return original