# 🧠 Smart Vision Analyzer (Computer Vision Project)

## 📌 Overview
This project is a backend-only computer vision pipeline that processes images using multiple techniques including preprocessing, feature extraction, segmentation, clustering, and motion analysis.

## 🚀 Features
- Image preprocessing (Grayscale, Histogram Equalization, Blur)
- Edge Detection (Canny)
- Feature Extraction (HOG)
- Corner Detection (Harris)
- Image Segmentation (Threshold + Contours)
- Object Detection (Bounding Boxes)
- Clustering (K-Means)
- Optical Flow (Video motion analysis)

## 🛠️ Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn

## Folder Format

vision_project/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── utils/
│ ├── preprocess.py
│ ├── features.py
│ ├── segmentation.py
│ ├── motion.py
│
├── inputs/ # User input images (any format)
├── outputs/ # Generated output images

## 🚀 Features
- Image preprocessing (Grayscale, Histogram Equalization, Blur)
- Edge Detection (Canny)

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
