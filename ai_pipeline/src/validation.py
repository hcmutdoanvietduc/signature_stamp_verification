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