import os
import cv2
import shutil
from config.settings import *
from src.preprocessing import *
from src.detection import *
from src.segmentation import *
# from src.validation import *
from src.utils import *

def setup_dirs():
    clear_folder(OUTPUT_DIR)
    """Tạo các thư mục output cần thiết nếu chưa tồn tại"""
    dirs = [
        PREPROCESSED_DOC_DIR,
        PREPROCESSED_STAMP_DIR,
        PREPROCESSED_SIGNATURE_DIR,
        OUTPUT_DETECTION_DIR,
        OVERLAY_CROP_DIR,
        SIGNATURE_CROP_DIR,
        STAMP_CROP_DIR,
        OUTPUT_SEGMENTATION_DIR,
        MASK_SIGNATURE_DIR,
        MASK_STAMP_DIR,
        MASK_OVERLAY_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Initialized output directories.")

def main():
    # 0a. Chuẩn bị thư mục
    setup_dirs()
    
    # 0b. Lấy danh sách văn bản đầu vào
    if not os.path.exists(INPUT_DOC_DIR):
        print(f"Input directory not found: {INPUT_DOC_DIR}")
        return

    input_files = [f for f in os.listdir(INPUT_DOC_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not input_files:
        print("No input documents found.")
        return

    # 1. Tiền xử lí văn bản
    print("Preprocessing documents...")

    batch_preprocess_documents(
        input_folder=INPUT_DOC_DIR,
        output_folder=PREPROCESSED_DOC_DIR,
        size=DOC_IMG_SIZE,
        deskew_threshold=8.0
    )

    # 2. Tiền xử lí CSDL chữ kí và con dấu
    print("Preprocessing signature and stamp databases...")

    create_database_masks(
        input_dir=DB_SIGNATURE_DIR,
        output_dir=PREPROCESSED_SIGNATURE_DIR,
        output_size=MASK_SIZE,
        min_component_size=15
    )

    create_database_masks(
        input_dir=DB_STAMP_DIR,
        output_dir=PREPROCESSED_STAMP_DIR,
        output_size=MASK_SIZE,
        min_component_size=15
    )

    # 3. Phát hiện chữ kí và con dấu trong văn bản
    print("Detecting signatures and stamps in documents...")

    for i in range(len(input_files)):
        detect_and_crop(
            input_image_path=os.path.join(PREPROCESSED_DOC_DIR, input_files[i]),
            overlay_dir=OVERLAY_CROP_DIR,
            stamp_dir=STAMP_CROP_DIR,
            signature_dir=SIGNATURE_CROP_DIR,
            model_path=YOLO_MODEL_PATH
        )
    
    # 4a. Dự đoán ảnh mask của vùng chồng lấn
    print("Predicting overlay masks...")

    predict_overlay_images_into_masks(
        overlay_dir=OVERLAY_CROP_DIR,
        model_path=UNET_MODEL_PATH,
        output_dir=MASK_OVERLAY_DIR
    )

    # 4b. Phân tách chữ kí và con dấu từ ảnh chồng lấn sử dụng ảnh mask
    print("Segmenting signatures and stamps from overlay masks...")

    signature_stamp_segmentation(
        overlay_mask_dir=MASK_OVERLAY_DIR,
        output_signature_dir=MASK_SIGNATURE_DIR,
        output_stamp_dir=MASK_STAMP_DIR
    )

    # 4c. Chuyển đổi vùng chữ kí/con dấu riêng biệt sang ảnh mask
    print("Generating specific signature and stamp masks...")

    create_masks(
        input_dir=SIGNATURE_CROP_DIR,
        output_dir=MASK_SIGNATURE_DIR,
        output_size=MASK_SIZE,
        min_component_size=15
    )

    create_masks(
        input_dir=STAMP_CROP_DIR,
        output_dir=MASK_STAMP_DIR,
        output_size=MASK_SIZE,
        min_component_size=15
    )

    # 5. Xác thực chữ kí và con dấu có trong CSDL hay không
    print("Validating signatures and stamps...")
    
    print("\n--- Pipeline Completed ---")

if __name__ == "__main__":
    main()