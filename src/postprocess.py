import numpy as np
import cv2

# Intentar importar skimage (opcional)
try:
    from skimage.morphology import (
        remove_small_objects, remove_small_holes,
        binary_opening, binary_closing, disk
    )
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


def clean_mask(mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    """
    Limpieza robusta de máscara binaria proveniente de segmentación.
    
    - Convierte a binaria (0/1)
    - Elimina pequeñas regiones ruidosas
    - Elimina pequeños huecos dentro de la grieta
    - Aplica apertura y cierre morfológico
    - Si skimage no está disponible, usa OpenCV
    """

    # Convertir valores a booleanos
    mask = (mask > 0).astype(np.uint8)

    # Si no hay nada, devolver vacío
    if np.sum(mask) < 1:
        return mask

    # ----------------------------
    # 🅰️ PROCESAMIENTO CON SKIMAGE
    # ----------------------------
    if SKIMAGE_AVAILABLE:
        mask_bool = mask.astype(bool)

        mask_bool = remove_small_objects(mask_bool, min_size=min_size)
        mask_bool = remove_small_holes(mask_bool, area_threshold=min_size)

        mask_bool = binary_opening(mask_bool, footprint=disk(2))
        mask_bool = binary_closing(mask_bool, footprint=disk(3))

        mask_clean = mask_bool.astype(np.uint8)

        if np.sum(mask_clean) > 0:
            return mask_clean

    # ----------------------------
    # 🅱️ ALTERNATIVA CON OPENCV (rápida)
    # ----------------------------

    # Eliminar ruido pequeño por conectividad
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned[labels == i] = 1

    # Apertura → eliminar puntos sueltos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Cierre → unir regiones
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2)

    return cleaned.astype(np.uint8)

