import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def _skeletonize_cv(mask: np.ndarray) -> np.ndarray:
    """
    Skeleton simple usando thinning de Zhang-Suen (OpenCV).
    Más rápido y estable que skimage.skeletonize en servidores.
    """
    mask = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = np.zeros_like(mask)

    while True:
        eroded = cv2.erode(mask, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(mask, opened)
        skel = cv2.bitwise_or(skel, temp)
        mask = eroded.copy()

        if cv2.countNonZero(mask) == 0:
            break

    return (skel > 0).astype(np.uint8)


def measure_crack(mask: np.ndarray, pixel_to_mm: float = 1.0):
    """
    Mide ancho, longitud, área y orientación de una grieta segmentada.
    Compatible con Render y entornos CPU.
    """

    mask = (mask > 0).astype(np.uint8)

    if np.sum(mask) < 10:
        # Máscara vacía o ruido
        return {
            "mean_width_mm": 0.0,
            "max_width_mm": 0.0,
            "length_mm": 0.0,
            "area_mm2": 0.0,
            "orientation_deg": 0.0
        }

    # 1️⃣ Esqueleto (línea central)
    skeleton = _skeletonize_cv(mask)

    if np.sum(skeleton) < 5:
        return {
            "mean_width_mm": 0.0,
            "max_width_mm": 0.0,
            "length_mm": 0.0,
            "area_mm2": float(np.sum(mask) * pixel_to_mm ** 2),
            "orientation_deg": 0.0
        }

    # 2️⃣ Distancia al borde
    dist = distance_transform_edt(mask)

    widths = dist[skeleton > 0] * 2 * pixel_to_mm
    if len(widths) == 0:
        mean_width = max_width = 0
    else:
        mean_width = float(np.mean(widths))
        max_width = float(np.max(widths))

    # 3️⃣ Área
    area = float(np.sum(mask) * pixel_to_mm ** 2)

    # 4️⃣ Longitud (aprox)
    length = float(np.sum(skeleton) * pixel_to_mm)

    # 5️⃣ Orientación por PCA robusto
    coords = np.column_stack(np.where(skeleton > 0))
    orientation = 0.0

    if len(coords) > 20:
        # PCA: eje principal
        coords_mean = np.mean(coords, axis=0)
        X = coords - coords_mean
        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        principal_axis = eigvecs[:, np.argmax(eigvals)]
        orientation = float(np.degrees(np.arctan2(principal_axis[0], principal_axis[1])))

    return {
        "mean_width_mm": mean_width,
        "max_width_mm": max_width,
        "length_mm": length,
        "area_mm2": area,
        "orientation_deg": orientation
    }
