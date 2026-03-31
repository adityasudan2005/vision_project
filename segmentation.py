import cv2
import numpy as np
from sklearn.cluster import KMeans


# Threshold Segmentation
def segment_image(img):
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    return thresh, contours


# Object Detection using Contours
def detect_objects(original, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 30 and h > 30:
            cv2.rectangle(
                original,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    return original


# K-Means Clustering
def apply_kmeans(img):
    data = img.reshape((-1, 1))

    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(data)

    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered = clustered.reshape(img.shape).astype(np.uint8)

    return clustered