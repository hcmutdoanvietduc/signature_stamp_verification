import os
from pathlib import Path

# Định nghĩa đường dẫn gốc
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn Model
MODEL_DIR = os.path.join(BASE_DIR, 'models')
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'best.pt')
UNET_MODEL_PATH = os.path.join(MODEL_DIR, 'best_unet_segmentation.keras')
SIAMESE_MODEL_PATH = os.path.join(MODEL_DIR, 'siamese_best.pth')

# Đường dẫn Data
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DOC_DIR = os.path.join(DATA_DIR, 'input_documents')
DB_SIGNATURE_DIR = os.path.join(DATA_DIR, 'db_signatures')
DB_STAMP_DIR = os.path.join(DATA_DIR, 'db_stamps')
INPUT_SIGNATURE_DIR = os.path.join(DATA_DIR, 'input_signatures')

OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Các thư mục con trong Output
PREPROCESSED_DOC_DIR = os.path.join(OUTPUT_DIR, 'input_preprocessed_docs')
PREPROCESSED_STAMP_DIR = os.path.join(OUTPUT_DIR, 'input_preprocessed_stamps')
PREPROCESSED_SIGNATURE_DIR = os.path.join(OUTPUT_DIR, 'input_preprocessed_signatures')

OUTPUT_DETECTION_DIR = os.path.join(OUTPUT_DIR, 'output_detection')
OVERLAY_CROP_DIR = os.path.join(OUTPUT_DIR, 'output_detection', 'overlay')
SIGNATURE_CROP_DIR = os.path.join(OUTPUT_DIR, 'output_detection', 'signature')
STAMP_CROP_DIR = os.path.join(OUTPUT_DIR, 'output_detection', 'stamp')

OUTPUT_SEGMENTATION_DIR = os.path.join(OUTPUT_DIR, 'output_segmentation')
MASK_OVERLAY_DIR = os.path.join(OUTPUT_SEGMENTATION_DIR, 'masks_overlay')
MASK_SIGNATURE_DIR = os.path.join(OUTPUT_SEGMENTATION_DIR,  'masks_signature')
MASK_STAMP_DIR = os.path.join(OUTPUT_SEGMENTATION_DIR, 'masks_stamp')

# Tham số
DOC_IMG_SIZE = (640, 640)
CROP_SIZE = (256, 256) # Kích thước crop overlay để đưa vào U-Net
MASK_SIZE = (128, 128) # Kích thước mask cuối cùng và input Siamese

SIGNATURE_IDS = {0}
STAMP_IDS = {1}
LABELS_BY_ID = {0: 'chu ky', 1: 'con dau'}

CONF_THRES_SIGNATURE = 0.20
CONF_THRES_STAMP = 0.20

BASE_CONF_THRES = min(CONF_THRES_SIGNATURE, CONF_THRES_STAMP)
NMS_IOU = 0.7
VERBOSE = False

touch_counts_as_overlap = False
MIN_IOU = 0.05
MIN_IOM = 0.15
MIN_IOA_STAMP = 0.10
MIN_IOA_SIGN = 0.10
PRIMARY_METRIC = 'iou'
CONF_BONUS_WEIGHT = 0.05
CROP_PAD_FACTOR = 0.05

DRAW_FINAL_ANNOTATION = True #optional
DRAW_LABELS_IN_ANNOTATION = True #optional
DRAW_UNION_LABELS = True #optional
SHOW_IN_COLAB = True #optional

COLOR_STAMP = (0, 255, 0)
COLOR_SIGN = (255, 0, 0)
COLOR_UNION = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)

PRINT_DETECTIONS = True #optional
PRINT_PAIR_SUMMARY = True #optional

MODEL_IN_SIZE = 256

BACKGROUND_VALUE = 0
SIGNATURE_VALUE = 128
STAMP_VALUE = 255
OVERLAP_VALUE = 192

TOL = 3

