import cv2

def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)

    # Gaussian Blur
    blur = cv2.GaussianBlur(hist_eq, (5, 5), 0)

    return blur