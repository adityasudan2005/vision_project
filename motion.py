import cv2
import numpy as np


def optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("Error reading video")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        print("Motion magnitude:", np.mean(mag))

        prvs = next_frame

    cap.release()