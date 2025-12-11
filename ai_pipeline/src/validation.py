import os, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SignatureEmbeddingNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 16x16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x  # [batch, emb_dim]


class SiameseNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.embedding = SignatureEmbeddingNet(emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # logit for "same person"
        )

    def forward(self, x1, x2):
        e1 = self.embedding(x1)
        e2 = self.embedding(x2)
        diff = torch.abs(e1 - e2)
        logits = self.classifier(diff)
        return logits, e1, e2

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

device = torch.device("cpu")

def same_person(model_path, img_path1, img_path2, threshold=0.5):
    model = SiameseNet(emb_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    img1 = Image.open(img_path1).convert("L")
    img2 = Image.open(img_path2).convert("L")
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _, _ = model(img1, img2)
        prob = torch.sigmoid(logits).item()
    return prob >= threshold

import os

def compare_with_db(model_path, input_img_path, db_root):
    """
    So sánh 1 ảnh (input_img_path) với toàn bộ DB (db_root: nhiều folder con).
    Trả về:
        - mã thư mục (vd: '01', 'A03', ...) nếu khớp với 1 người trong DB
        - None nếu không khớp ai
    """
    for subfolder in sorted(os.listdir(db_root)):
        subfolder_path = os.path.join(db_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        img_files = [
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not img_files:
            continue

        # có thể lặp hết img_files; ở đây demo dùng từng file
        for img_name in img_files:
            db_img_path = os.path.join(subfolder_path, img_name)

            # same_person(model_path, query, candidate) -> True/False
            is_same = same_person(model_path, input_img_path, db_img_path)

            if is_same:
                # input_img thuộc về signer / stamp có id = subfolder
                return subfolder

    # không ai khớp trong DB
    return None

def compare_test_folder_with_db(model_path, mask_folder, db_root):
    """
    mask_folder: folder chứa mask sau khi đã segmentation (bây giờ là dạng flat: 01_1_in_db.jpg, ...)
    db_root:   folder DB, vẫn dạng nhiều folder con theo signer
    """
    results = {}

    for fname in os.listdir(mask_folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_img_path = os.path.join(mask_folder, fname)
        owner_id = compare_with_db(model_path, input_img_path, db_root)

        results[fname] = owner_id
        if owner_id is None:
            print(f"{fname} -> UNKNOWN")
        else:
            print(f"{fname} -> {owner_id}")

    return results

    return results


