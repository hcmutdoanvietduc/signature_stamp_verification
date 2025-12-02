import os, shutil
import cv2
import numpy as np
from glob import glob
# import tensorflow as tf
from src.preprocessing import *
from config.settings import *

#Helper functions for preprocessing
def batch_preprocess_documents(input_folder, output_folder,
                            size=(640, 640),
                            deskew_folder=None,
                            deskew_threshold=5.0):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    files = []
    for e in exts:
        files.extend(glob(os.path.join(input_folder, e)))

    if not files:
        print(" Không tìm thấy ảnh trong thư mục đầu vào.")
        return

    print(f"Tìm thấy {len(files)} ảnh. Bắt đầu xử lý...\n")

    for img_path in files:
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_folder, filename)
        preprocess_document_image(img_path, out_path,
                                  size=size,
                                  deskew_folder=deskew_folder,
                                  deskew_threshold=deskew_threshold)

    print("\nHoàn thành xử lý tất cả ảnh!")

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Đã xóa thư mục và nội dung: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)
    print(f"Đã tạo lại thư mục: {folder_path}")

# Helper functions for detection
def tlwh_from_xyxy(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def sanitize_xyxy(box, W, H, as_int=True):
    x1, y1, x2, y2 = [float(b) for b in box]
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = max(0.0, min(x1, float(W)))
    y1 = max(0.0, min(y1, float(H)))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))
    if as_int:
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return [x1, y1, x2, y2]

def intersect_xyxy(a, b, touch_ok=False, as_int=False):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    cond = (x2 >= x1 and y2 >= y1) if touch_ok else (x2 > x1 and y2 > y1)
    if not cond: return None
    if as_int: return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return [float(x1), float(y1), float(x2), float(y2)]

def union_xyxy(a, b, as_int=False):
    x1 = min(a[0], b[0]); y1 = min(a[1], b[1])
    x2 = max(a[2], b[2]); y2 = max(a[3], b[3])
    if as_int: return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return [float(x1), float(y1), float(x2), float(y2)]

def area_xyxy(b):
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

def iou_xyxy(a, b):
    inter = intersect_xyxy(a, b, touch_ok=False, as_int=False)
    if inter is None: return 0.0
    inter_area = area_xyxy(inter)
    ua = area_xyxy(a) + area_xyxy(b) - inter_area
    return inter_area / ua if ua > 0 else 0.0

def iom_xyxy(a, b):
    inter = intersect_xyxy(a, b, touch_ok=False, as_int=False)
    if inter is None: return 0.0
    inter_area = area_xyxy(inter)
    denom = min(area_xyxy(a), area_xyxy(b))
    return inter_area / denom if denom > 0 else 0.0

def pad_box_xyxy(box, pad_factor, W, H, as_int=True):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    dx = w * float(pad_factor); dy = h * float(pad_factor)
    x1 -= dx; y1 -= dy; x2 += dx; y2 += dy
    return sanitize_xyxy([x1, y1, x2, y2], W, H, as_int=as_int)

def accept_pair_by_thresholds(s_box, g_box):
    inter = intersect_xyxy(s_box, g_box, touch_ok=touch_counts_as_overlap, as_int=False)
    if inter is None:
        return False, inter, 0.0, 0.0, 0.0, 0.0
    inter_area = area_xyxy(inter)
    if not touch_counts_as_overlap and inter_area <= 0:
        return False, inter, 0.0, 0.0, 0.0, 0.0
    iou = iou_xyxy(s_box, g_box)
    iom = iom_xyxy(s_box, g_box)
    ioa_stamp = inter_area / area_xyxy(s_box) if area_xyxy(s_box) > 0 else 0.0
    ioa_sign = inter_area / area_xyxy(g_box) if area_xyxy(g_box) > 0 else 0.0
    ok = (iou >= MIN_IOU) or (iom >= MIN_IOM) or ((ioa_stamp >= MIN_IOA_STAMP) and (ioa_sign >= MIN_IOA_SIGN))
    return ok, inter, iou, iom, ioa_stamp, ioa_sign

def draw_box_with_label(img, box_xyxy, color, label_text=None, thickness=2, font_scale=0.5, text_thickness=1):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label_text:
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad = 3
        tx1, ty1 = x1, max(0, y1 - th - baseline - 2*pad)
        tx2, ty2 = x1 + tw + 2*pad, y1
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), COLOR_TEXT_BG, thickness=-1)
        cv2.putText(img, label_text, (x1 + pad, y1 - baseline - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, text_thickness, lineType=cv2.LINE_AA)

def save_crop(img, box_xyxy, out_path, pad_factor=0.0):
    H, W = img.shape[:2]
    box = sanitize_xyxy(box_xyxy, W, H, as_int=True)
    if pad_factor and pad_factor > 0:
        box = pad_box_xyxy(box, pad_factor, W, H, as_int=True)
    x1, y1, x2, y2 = box
    if (x2 - x1) > 0 and (y2 - y1) > 0:
        crop = img[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, CROP_SIZE)
        ok = cv2.imwrite(out_path, crop_resized)
        return ok
    return False

#Helper functions for segmentation
def preprocess_for_model(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    return img

def preds_to_classmap(pred):
    if pred.ndim == 3 and pred.shape[-1] > 1:
        classmap = np.argmax(pred, axis=-1).astype(np.uint8)
    else:
        classmap = np.squeeze(pred)
        classmap = np.rint(classmap).astype(np.uint8)
    return classmap

def classmap_to_visual_mask(classmap, mapping=(0,1,2,3)):
    vis = np.zeros_like(classmap, dtype=np.uint8)
    vis[classmap == 0] = BACKGROUND_VALUE
    vis[classmap == 1] = SIGNATURE_VALUE
    vis[classmap == 2] = STAMP_VALUE
    vis[classmap == 3] = OVERLAP_VALUE
    return vis

def save_rgba_with_alpha(fg_bgr, mask_binary, out_path):
    b,g,r = cv2.split(fg_bgr)
    a = mask_binary
    rgba = cv2.merge([b,g,r,a])
    cv2.imwrite(out_path, rgba)

# def to_uint8_gray(img):
#     if img.ndim == 3:
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#     else:
#         img_gray = img

#     if img_gray.dtype != np.uint8:
#         if img_gray.max() <= 1.0:
#             img_gray = np.rint(img_gray * 255).astype(np.uint8)
#         else:
#             img_gray = img_gray.astype(np.uint8)
#     return img_gray

def to_uint8_gray(img):
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            img_gray = np.squeeze(img, axis=2)
        else:
            img_gray = img
    else:
        img_gray = img

    if img_gray.dtype != np.uint8:
        if img_gray.max() <= 1.0:
            img_gray = np.rint(img_gray * 255).astype(np.uint8)
        else:
            img_gray = img_gray.astype(np.uint8)
            
    return img_gray

def in_range(x, v, tol=TOL):
    return (x >= v - tol) & (x <= v + tol)

def to_binary_uint8(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return ((mask > 0).astype(np.uint8) * 255)

def resize_mask_binary(mask, size=(128, 128)):
    mask = to_binary_uint8(mask)
    resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    resized = to_binary_uint8(resized)
    return resized