# utils.py

import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from skimage.morphology import skeletonize


# ============================================================
# Datasets
# ============================================================

class CrackClassificationDataset(Dataset):
    """
    Estructura esperada:
      /Positive/*.jpg  -> label = 1
      /Negative/*.jpg  -> label = 0
    """
    def __init__(self, root_dir, transform=None):
        pos = sorted(glob.glob(os.path.join(root_dir, 'Positive', '*')))
        neg = sorted(glob.glob(os.path.join(root_dir, 'Negative', '*')))

        self.images = pos + neg
        self.labels = [1]*len(pos) + [0]*len(neg)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, label, os.path.basename(path)



class CrackSegmentationDataset(Dataset):
    """
    Estructura esperada:
      /images/*.jpg/png
      /masks/*.png
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(images_dir, '*')))
        mask_list = sorted(glob.glob(os.path.join(masks_dir, '*')))

        mask_map = {os.path.splitext(os.path.basename(x))[0]: x for x in mask_list}

        # Emparejar
        self.paired = []
        for img in self.images:
            key = os.path.splitext(os.path.basename(img))[0]
            if key in mask_map:
                self.paired.append((img, mask_map[key]))

        self.transform = transform

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        img_path, mask_path = self.paired[idx]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = (np.array(mask) > 127).astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask, os.path.basename(img_path)



# ============================================================
# Métricas
# ============================================================

def iou_pytorch(outputs, labels, smooth=1e-6):
    outputs = (outputs > 0.5).float()
    intersection = (outputs * labels).sum(dim=(1,2,3))
    union = outputs.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


def compute_classification_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }



# ============================================================
# Medición de fisuras (core)
# ============================================================

def crack_widths_from_mask(mask_bin):
    """
    Retorna:
      widths: lista de anchos locales en píxeles
      summary: mean, median, max
    """
    mask_bin = (mask_bin > 0).astype(np.uint8)

    if mask_bin.sum() == 0:
        return [], {"mean": 0, "median": 0, "max": 0}

    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    skel = skeletonize(mask_bin).astype(np.uint8)

    widths = dist[skel == 1] * 2

    if len(widths) == 0:
        return [], {"mean": 0, "median": 0, "max": 0}

    return widths.tolist(), {
        "mean": float(np.mean(widths)),
        "median": float(np.median(widths)),
        "max": float(np.max(widths)),
    }


def convert_pixels_to_mm(value_px, px_per_mm):
    """
    value_px: número en píxeles
    px_per_mm: cuántos píxeles equivalen a 1 mm
    """
    if px_per_mm is None or px_per_mm <= 0:
        return None
    return round(value_px / px_per_mm, 3)



# ============================================================
# Funciones adicionales
# ============================================================

def compute_pos_weight(masks_dir):
    """
    Calcula pos_weight = (#negativos / #positivos)
    útil para BCEWithLogitsLoss.
    """
    masks = glob.glob(os.path.join(masks_dir, '*'))
    pos = 0
    neg = 0

    for m in masks:
        arr = np.array(Image.open(m).convert('L'))
        bin_mask = (arr > 127).astype(np.uint8)

        pos += bin_mask.sum()
        neg += (bin_mask.size - bin_mask.sum())

    if pos == 0:
        return 1.0  # prevenir división por cero

    return float(neg) / float(pos)



def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
