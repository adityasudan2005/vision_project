import cv2
import os
from utils.preprocess import preprocess_image
from utils.features import detect_edges, extract_hog, detect_corners
from utils.segmentation import segment_image, detect_objects, apply_kmeans
from utils.motion import optical_flow


def print_step(step):
    print(f"\n🔹 {step}")


def run_pipeline(image_path):
    print("\n📌 Starting Computer Vision Pipeline...\n")

    # Check file exists
    if not os.path.exists(image_path):
        print("❌ ERROR: File not found. Please check the path.")
        return

    # Load image (supports all formats OpenCV supports)
    img = cv2.imread(image_path)

    if img is None:
        print("❌ ERROR: Unsupported or corrupted image file.")
        return

    print("✅ Image loaded successfully!")
    print(f"📐 Image shape: {img.shape}")

    # Step 1: Preprocessing
    print_step("Preprocessing (Grayscale + Histogram Equalization + Blur)")
    pre = preprocess_image(img)

    # Step 2: Edge Detection
    print_step("Edge Detection (Canny)")
    edges = detect_edges(pre)
    print(f"✔ Edges detected. Shape: {edges.shape}")

    # Step 3: Feature Extraction
    print_step("Feature Extraction (HOG)")
    hog_features = extract_hog(pre)
    print(f"✔ HOG feature vector length: {len(hog_features)}")

    # Step 4: Corner Detection
    print_step("Corner Detection (Harris)")
    corners_img = detect_corners(pre, img.copy())
    print("✔ Corners highlighted in red")

    # Step 5: Segmentation
    print_step("Image Segmentation (Threshold + Contours)")
    segmented, contours = segment_image(pre)
    print(f"✔ Total contours detected: {len(contours)}")

    # Step 6: Object Detection
    print_step("Object Detection (Bounding Boxes from Contours)")
    detected = detect_objects(img.copy(), contours)
    print("✔ Bounding boxes drawn")

    # Step 7: Clustering
    print_step("Clustering (K-Means)")
    clustered = apply_kmeans(pre)
    print("✔ Image clustered into regions")

    # Save outputs
    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite("outputs/edges.png", edges)
    cv2.imwrite("outputs/corners.png", corners_img)
    cv2.imwrite("outputs/segmented.png", segmented)
    cv2.imwrite("outputs/detected.png", detected)
    cv2.imwrite("outputs/clustered.png", clustered)

    print("\n📁 Outputs saved in 'outputs/' folder:")
    print("   - edges.png")
    print("   - corners.png")
    print("   - segmented.png")
    print("   - detected.png")
    print("   - clustered.png")

    print("\n🎉 Pipeline Execution Completed Successfully!\n")


# =========================
# USER INPUT SECTION
# =========================
if __name__ == "__main__":
    while True:
        print("=== 🧠 Smart Vision Analyzer ===")

        print("1. Process an Image")
        print("2. Exit")
        choice = input("Select an option (1 or 2): ").strip()
        if choice == '1':
            image_path = input("\n📂 Enter full image path (any format): ").strip()
            run_pipeline(image_path)
        elif choice == '2':
            print("👋 Exiting. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1 or 2.\n") 