import streamlit as st
import os
import tempfile
from PIL import Image
import torch

# -----------------------------------------------
# 📥 Descarga de modelos (solo si faltan)
# -----------------------------------------------
try:
    from models.download_models import download_models
    download_models()
except Exception as e:
    print(f"[WARN] No se pudo ejecutar download_models(): {e}")

# -----------------------------------------------
# 📦 Import correcto desde carpeta src/
# -----------------------------------------------
from src.inference import inference


# -----------------------------------------------
# ⚙️ Configuración general Streamlit
# -----------------------------------------------
st.set_page_config(
    page_title="Evaluación de Fisuras con Deep Learning",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Sistema de Evaluación de Fisuras con Deep Learning")
st.markdown("""
Esta aplicación permite **detectar, segmentar y evaluar el riesgo** de fisuras en edificaciones.
Puedes **subir una foto** o **capturar una con tu cámara** 📸.
""")

# -----------------------------------------------
# 📁 Asegurar carpetas de salida
# -----------------------------------------------
os.makedirs("outputs/inference", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# -----------------------------------------------
# 📤 Subir imagen
# -----------------------------------------------
uploaded_file = st.file_uploader("📁 Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

# -----------------------------------------------
# 📸 Foto desde cámara
# -----------------------------------------------
camera_image = st.camera_input("📸 O toma una foto con tu cámara")

# -----------------------------------------------
# 🔍 Selección final
# -----------------------------------------------
img_source = uploaded_file or camera_image

if img_source:
    st.markdown("---")
    st.subheader("🔍 Resultados del análisis")

    # Guardar temporalmente la imagen
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_source.getvalue())
        temp_path = tmp.name

    # Mostrar imagen original
    st.image(temp_path, caption="Imagen seleccionada", use_column_width=True)

    # -------------------------------------------
    # 🧠 Ubicación de modelos en producción
    # -------------------------------------------
    cls_ckpt = "models/cls_best.pth"
    seg_ckpt = "models/seg_best.pth"

    # Validar existencia
    if not os.path.exists(cls_ckpt):
        st.error("❌ FALTA el modelo de clasificación (cls_best.pth).")
        st.stop()

    if not os.path.exists(seg_ckpt):
        st.error("❌ FALTA el modelo de segmentación (seg_best.pth).")
        st.stop()

    # -------------------------------------------
    # 🚀 Inferencia
    # -------------------------------------------
    try:
        with st.spinner("Analizando imagen... 🧠"):
            results = inference(
                image_path=temp_path,
                cls_ckpt=cls_ckpt,
                seg_ckpt=seg_ckpt,
                output_dir="outputs/inference",
                img_size=512,
                device="cpu"  # Render no tiene GPU
            )
    except Exception as e:
        st.error("❌ Error durante la inferencia.")
        st.exception(e)
        st.stop()

    # -------------------------------------------
    # 📊 Mostrar resultados
    # -------------------------------------------
    if not results["fisura"]:
        st.success("✅ No se detectaron fisuras.")
    else:
        st.image(
            results["mask_path"],
            caption=f"Overlay — Nivel de riesgo: {results['riesgo']}",
            use_column_width=True
        )

        m = results["medidas"]
        st.markdown(f"""
        ### 📊 Medidas detectadas  
        *(pixel_to_mm = 1.0 por defecto)*  

        - **Ancho medio:** `{m['mean_width_mm']:.3f} mm`  
        - **Ancho máximo:** `{m['max_width_mm']:.3f} mm`  
        - **Longitud:** `{m['length_mm']:.3f} mm`  
        - **Área:** `{m['area_mm2']:.3f} mm²`  
        - **Orientación:** `{m['orientation_deg']:.1f}°`  
        - 🧭 **Riesgo final: {results['riesgo']}**
        """)

    st.markdown("---")
    st.caption("💡 Consejo: usa imágenes claras, cercanas y bien iluminadas.")

else:
    st.info("📌 Sube o captura una imagen para comenzar.")
