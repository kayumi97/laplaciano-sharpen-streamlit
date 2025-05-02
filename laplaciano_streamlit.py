import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="centered")
st.title("🔎 Realce com Laplaciano")

# Upload da imagem
uploaded_file = st.file_uploader("📤 Envie uma imagem (JPG ou PNG)", type=["jpg", "png"])
if uploaded_file:
    # Leitura da imagem com OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Slider para controlar o alpha
    alpha = st.slider("💡 Intensidade do Laplaciano (alpha)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    # Aplicar Laplaciano
    laplaciano = cv2.Laplacian(img_gray, cv2.CV_64F)
    img_float = img_gray.astype(np.float64)
    img_realcada = img_float - alpha * laplaciano
    img_realcada = np.clip(img_realcada, 0, 255).astype(np.uint8)

    # Mostrar as imagens
    col1, col2, col3 = st.columns(3)
    col1.image(img_gray, caption="Imagem Original", use_container_width=True, channels="GRAY")
    col2.image(cv2.convertScaleAbs(laplaciano), caption="Laplaciano", use_container_width=True, channels="GRAY")
    col3.image(img_realcada, caption="Imagem Realçada", use_container_width=True, channels="GRAY")


    # Download opcional
    img_pil = Image.fromarray(img_realcada)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("💾 Baixar imagem realçada", data=byte_im, file_name="realcada.png", mime="image/png")
