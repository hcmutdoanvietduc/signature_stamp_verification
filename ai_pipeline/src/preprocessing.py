import cv2
import numpy as np
import os

#Calculate skew angle of the document
def compute_skew_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:  # chỉ lấy góc gần ngang
            angles.append(angle)

    if len(angles) == 0:
        return 0.0

    median_angle = float(np.median(angles))
    return median_angle

# Preprocess document image: deskew, crop, resize, save
def preprocess_document_image(input_path, output_path,
                              size=(640, 640),
                              deskew_folder=None,
                              deskew_threshold=5.0):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Không đọc được ảnh: {input_path}")
        return None

    # --- Deskew ---
    angle = compute_skew_angle(image)
    rotated = False
    if abs(angle) > deskew_threshold:  # chỉ xoay khi lệch rõ
        rotated = True
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    else:
        angle = 0.0

    # --- Crop lề trắng ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        image = image[y:y+h, x:x+w]

    # --- Resize về 640x640 ---
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # --- Lưu ra JPG ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_jpg = os.path.splitext(output_path)[0] + ".jpg"
    cv2.imwrite(output_jpg, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return output_jpg

# Create binary masks for databases of signature/stamp
def create_database_masks(input_dir, output_dir,
                          output_size=(128, 128),
                          min_component_size=20):
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return

    for subdir in subdirs:
        sub_input_path = os.path.join(input_dir, subdir)
        sub_output_path = os.path.join(output_dir, subdir)
        
        os.makedirs(sub_output_path, exist_ok=True)
        
        create_masks(sub_input_path, sub_output_path, output_size, min_component_size)
        
# Create binary masks for folders of signature/stamp
def create_masks(input_dir, output_dir, output_size=(128, 128), min_component_size=20):
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                cleaned_mask[labels == i] = 255

        mask_resized = cv2.resize(cleaned_mask, output_size, interpolation=cv2.INTER_NEAREST)

        mask_name = os.path.splitext(img_file)[0] + ".png"
        mask_path = os.path.join(output_dir, mask_name)
        cv2.imwrite(mask_path, mask_resized)
    
