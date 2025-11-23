import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
from torch.amp import autocast
import warnings

# Silenciar future warnings seguro
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------
# IMPORTS INTERNOS (rutas corregidas para producción)
# ---------------------------------------------------
from src.postprocess import clean_mask
from src.measure_crack import measure_crack as measure_crack_mm
from src.risk_classifier import classify_risk


# ---------------------------------------------------
# 🔧 Transformaciones para preprocesamiento
# ---------------------------------------------------
def get_transform(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ---------------------------------------------------
# 🧠 Clasificador ResNet18
# ---------------------------------------------------
def load_classifier(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ No se encontró modelo de clasificación en {model_path}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------
# 🧠 Segmentador DeepLabV3+
# ---------------------------------------------------
def load_segmenter(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ No se encontró modelo de segmentación en {model_path}")

    model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=False)
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)

    ckpt = torch.load(model_path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------
# 🔍 FUNCION PRINCIPAL DE INFERENCIA
# ---------------------------------------------------
def inference(image_path, cls_ckpt, seg_ckpt,
              output_dir="outputs/inference",
              img_size=512, pixel_to_mm=1.0,
              device=None):

    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Usando dispositivo: {device}")

    transform = get_transform(img_size)

    # 1️⃣ --- Cargar imagen ---
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # 2️⃣ --- Clasificación ---
    classifier = load_classifier(cls_ckpt, device)

    with torch.no_grad():
        if device == "cuda":
            with autocast("cuda", dtype=torch.float16):
                logits = classifier(tensor)
        else:
            logits = classifier(tensor)

    pred_class = torch.argmax(logits, dim=1).item()

    if pred_class == 0:
        return {
            "fisura": False,
            "mensaje": "No se detectan fisuras en la imagen."
        }

    # 3️⃣ --- Segmentación ---
    segmenter = load_segmenter(seg_ckpt, device)

    with torch.no_grad():
        if device == "cuda":
            with autocast("cuda", dtype=torch.float16):
                seg_out = segmenter(tensor)["out"]
        else:
            seg_out = segmenter(tensor)["out"]

    mask_prob = torch.sigmoid(seg_out).squeeze().cpu().numpy()

    # 4️⃣ --- Postprocesamiento ---
    mask_bin = (mask_prob > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_bin, (img.width, img.height), cv2.INTER_NEAREST)
    mask_clean = clean_mask(mask_resized)

    # 5️⃣ --- Medición ---
    measures = measure_crack_mm(mask_clean, pixel_to_mm)
    risk = classify_risk(measures["mean_width_mm"])

    # 6️⃣ --- Overlay ---
    base = np.array(img)
    overlay = base.copy()
    overlay[mask_clean.astype(bool)] = [255, 0, 0]

    blended = cv2.addWeighted(base, 0.7, overlay, 0.3, 0)

    # 7️⃣ --- Guardar resultados ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    out_img = os.path.join(output_dir, f"{base_name}_pred.png")
    Image.fromarray(blended).save(out_img)

    return {
        "fisura": True,
        "riesgo": risk,
        "medidas": measures,
        "mask_path": out_img
    }


# ---------------------------------------------------
# 🚀 MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--cls_ckpt", default="models/cls_best.pth")
    parser.add_argument("--seg_ckpt", default="models/seg_best.pth")
    parser.add_argument("--output_dir", default="outputs/inference")
    args = parser.parse_args()

    inference(
        args.image_path,
        args.cls_ckpt,
        args.seg_ckpt,
        args.output_dir
    )
