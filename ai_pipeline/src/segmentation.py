import tensorflow as tf
import cv2
import numpy as np
import os
from config.settings import *
from src.utils import *

# Predict overlay images into segmentation masks using U-Net model
def predict_overlay_images_into_masks(overlay_dir=OVERLAY_CROP_DIR, model_path=UNET_MODEL_PATH, 
                                      output_dir=MASK_OVERLAY_DIR):
    model = tf.keras.models.load_model(model_path, compile=False)
    image_files = [f for f in os.listdir(overlay_dir) if f.lower().endswith(('.png'))]

    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục: {overlay_dir}")
    else:
        for filename in image_files:
            full_image_path = os.path.join(overlay_dir, filename)
            orig = cv2.imread(full_image_path)
            if orig is None:
                print(f"Không thể đọc ảnh: {full_image_path}. Bỏ qua.")
                continue

            img_for_model = preprocess_for_model(orig)
            inp = np.expand_dims(img_for_model, axis=0)

            pred = model.predict(inp)[0]

            classmap_small = preds_to_classmap(pred)

            mask_small_vis = classmap_to_visual_mask(classmap_small)

            mask_full = mask_small_vis

            mask_filename = os.path.splitext(filename)[0] + "_mask.png"
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, mask_full)
            print(f"Saved mask for {filename}")

#Segment signatures and stamps from overlay masks
def signature_stamp_segmentation(overlay_mask_dir=MASK_OVERLAY_DIR, output_signature_dir=MASK_SIGNATURE_DIR, 
                                 output_stamp_dir=MASK_STAMP_DIR):
    images = []
    filenames = []

    for filename in os.listdir(overlay_mask_dir):
        filepath = os.path.join(overlay_mask_dir, filename)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                filenames.append(filename)

    print(f"Loaded {len(images)} images.")

    signature_masks = []
    stamp_masks = []
    overlap_masks = []

    for img in images:
        g = to_uint8_gray(img)

        signature_mask = (in_range(g, SIGNATURE_VALUE)     ).astype(np.uint8) * 255
        stamp_mask     = (in_range(g, STAMP_VALUE)         ).astype(np.uint8) * 255
        overlap_mask   = (in_range(g, OVERLAP_VALUE)       ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        signature_mask = cv2.morphologyEx(signature_mask, cv2.MORPH_CLOSE, kernel)
        stamp_mask     = cv2.morphologyEx(stamp_mask,     cv2.MORPH_CLOSE, kernel)
        overlap_mask   = cv2.morphologyEx(overlap_mask,   cv2.MORPH_CLOSE, kernel)

        signature_masks.append(signature_mask)
        stamp_masks.append(stamp_mask)
        overlap_masks.append(overlap_mask)

    print(f"Generated {len(signature_masks)} signature masks, {len(stamp_masks)} stamp masks, and {len(overlap_masks)} overlap masks.")

    final_signature_masks = []
    final_stamp_masks = []

    for i in range(len(signature_masks)):
        g = to_uint8_gray(images[i])
        final_signature_mask = np.isin(g, [SIGNATURE_VALUE, OVERLAP_VALUE]).astype(np.uint8) * 255
        final_stamp_mask     = np.isin(g, [STAMP_VALUE,     OVERLAP_VALUE]).astype(np.uint8) * 255

        final_signature_masks.append(final_signature_mask)
        final_stamp_masks.append(final_stamp_mask)

    print(f"Generated {len(final_signature_masks)} final signature masks and {len(final_stamp_masks)} final stamp masks.")

    resized_signature_masks = [resize_mask_binary(m, (128, 128)) for m in final_signature_masks]
    resized_stamp_masks     = [resize_mask_binary(m, (128, 128)) for m in final_stamp_masks]

    print(f"Resized {len(resized_signature_masks)} signature masks and {len(resized_stamp_masks)} stamp masks to 128x128.")

    for i, (signature_mask, stamp_mask) in enumerate(zip(resized_signature_masks, resized_stamp_masks)):
        original_filename = filenames[i]
        base_filename, _ = os.path.splitext(original_filename)

        signature_output_path = os.path.join(output_signature_dir, f"{base_filename}_signature_mask.png")
        stamp_output_path = os.path.join(output_stamp_dir, f"{base_filename}_stamp_mask.png")

        cv2.imwrite(signature_output_path, signature_mask)
        cv2.imwrite(stamp_output_path, stamp_mask)

    print(f"Saved {len(resized_signature_masks)} signature masks to {output_signature_dir} and {len(resized_stamp_masks)} stamp masks to {output_stamp_dir}")


