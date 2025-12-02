from ultralytics import YOLO
import cv2
import os
from config.settings import *
from src.utils import *

#Detect and crop function for each image of document
def detect_and_crop(input_image_path, overlay_dir=OVERLAY_CROP_DIR, stamp_dir=STAMP_CROP_DIR,
                     signature_dir=SIGNATURE_CROP_DIR, model_path=YOLO_MODEL_PATH):
    model = YOLO(model_path)

    image = cv2.imread(input_image_path)
    if image is None:
        raise RuntimeError(f"Lỗi: Không thể đọc ảnh từ đường dẫn {input_image_path}")

    image_clean = image.copy()
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    results = model(image, conf=BASE_CONF_THRES, iou=NMS_IOU, verbose=VERBOSE)

    detections = []
    stamp_dets = []
    sign_dets = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy[0]
            try:
                x1, y1, x2, y2 = xyxy.detach().cpu().numpy().tolist()
            except Exception:
                x1, y1, x2, y2 = xyxy.tolist()

            tlwh = tlwh_from_xyxy(x1, y1, x2, y2)
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            class_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)

            entry = {
                'class_id': cls_id,
                'class_name': class_name,
                'conf': conf,
                'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'tlwh': tlwh
            }
            detections.append(entry)

            if cls_id in STAMP_IDS and conf >= CONF_THRES_STAMP:
                stamp_dets.append(entry)
            elif cls_id in SIGNATURE_IDS and conf >= CONF_THRES_SIGNATURE:
                sign_dets.append(entry)

    H, W = image_clean.shape[:2]
    stamp_boxes_xyxy = [sanitize_xyxy(d['xyxy'], W, H, as_int=False) for d in stamp_dets]
    sign_boxes_xyxy = [sanitize_xyxy(d['xyxy'], W, H, as_int=False) for d in sign_dets]

    accepted_pairs = []   # list of dicts: {'s_idx','g_idx','union','inter','iou','iom','score','conf_min'}
    for i, s in enumerate(stamp_boxes_xyxy):
        for j, g in enumerate(sign_boxes_xyxy):
            ok, inter, iou, iom, ioa_s, ioa_g = accept_pair_by_thresholds(s, g)
            if not ok:
                continue
            u = union_xyxy(s, g, as_int=False)
            conf_min = min(stamp_dets[i]['conf'], sign_dets[j]['conf'])
            score = (iou if PRIMARY_METRIC == 'iou' else (iom if PRIMARY_METRIC == 'iom' else (2.0*area_xyxy(inter)/(area_xyxy(s)+area_xyxy(g)) if inter is not None else 0.0))) \
                    + CONF_BONUS_WEIGHT * conf_min
            accepted_pairs.append({
                's_idx': i, 'g_idx': j,
                'union': u, 'inter': inter,
                'iou': iou, 'iom': iom, 'score': score,
                'conf_min': conf_min
            })

    used_stamps = set()
    used_signs = set()

    overlay_count = 0
    for k, p in enumerate(accepted_pairs):
        u_box = sanitize_xyxy(p['union'], W, H, as_int=True)
        u_box = pad_box_xyxy(u_box, CROP_PAD_FACTOR, W, H, as_int=True)
        # save
        fname = f"{base_name}_overlay_{(overlay_count+1):02d}.png"
        out_path = os.path.join(overlay_dir, fname)
        ok = save_crop(image_clean, u_box, out_path, pad_factor=0.0)  # already padded
        if ok:
            overlay_count += 1
            used_stamps.add(p['s_idx'])
            used_signs.add(p['g_idx'])

    leftover_stamps = [idx for idx in range(len(stamp_boxes_xyxy)) if idx not in used_stamps]
    leftover_signs = [idx for idx in range(len(sign_boxes_xyxy)) if idx not in used_signs]

    stamp_count = 0
    for t, idx in enumerate(leftover_stamps):
        box = sanitize_xyxy(stamp_boxes_xyxy[idx], W, H, as_int=True)
        box = pad_box_xyxy(box, CROP_PAD_FACTOR, W, H, as_int=True)
        fname = f"{base_name}_stamp_{(stamp_count+1):02d}.png"
        out_path = os.path.join(stamp_dir, fname)
        ok = save_crop(image_clean, box, out_path, pad_factor=0.0)
        if ok:
            stamp_count += 1

    sign_count = 0
    for t, idx in enumerate(leftover_signs):
        box = sanitize_xyxy(sign_boxes_xyxy[idx], W, H, as_int=True)
        box = pad_box_xyxy(box, CROP_PAD_FACTOR, W, H, as_int=True)
        fname = f"{base_name}_signature_{(sign_count+1):02d}.png"
        out_path = os.path.join(signature_dir, fname)
        ok = save_crop(image_clean, box, out_path, pad_factor=0.0)
        if ok:
            sign_count += 1

    print(f"Finish {os.path.basename(input_image_path)}")


    