import os
import urllib.request

MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------
# 🔗 URLs directas de Google Drive
# (convertidas a descarga directa)
# ---------------------------------------
MODELS = {
    "cls_best.pth": "https://drive.google.com/uc?export=download&id=1f5HzoIo88GA2hqppnd9zkxsjHEHSJKrx",
    "seg_best.pth": "https://drive.google.com/uc?export=download&id=1qz_CKihk_WGoDbJQXRwdKrzEXWp9gMMG"
}

def safe_download(url, dst_path):
    """Descarga archivo con manejo seguro de errores."""
    try:
        print(f"[INFO] Descargando: {url}")
        urllib.request.urlretrieve(url, dst_path)
        print(f"[OK] Guardado en {dst_path}")

        # Validar tamaño mínimo (para detectar descargas corruptas)
        if os.path.getsize(dst_path) < 1000:
            raise ValueError("Archivo descargado demasiado pequeño, posible error.")

    except Exception as e:
        print(f"[ERROR] No se pudo descargar {url}: {e}")
        if os.path.exists(dst_path):
            os.remove(dst_path)
        raise e


def download_models():
    print("[INFO] Verificando modelos...")

    for filename, url in MODELS.items():
        path = os.path.join(MODEL_DIR, filename)

        # Si el archivo ya existe, no lo descargamos
        if os.path.exists(path):
            print(f"[OK] {filename} encontrado, no se descarga.")
            continue

        print(f"[INFO] {filename} no encontrado. Descargando...")
        safe_download(url, path)

    print("[INFO] Todos los modelos están listos.")


if __name__ == "__main__":
    download_models()
