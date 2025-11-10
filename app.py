# app.py
import json, os, io
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ----------- Ajustes UI -----------
st.set_page_config(
    page_title="Clasificación de Vehículos Militares",
    page_icon="🪖",
    layout="wide"
)

# Pequeño estilo para look limpio
st.markdown("""
<style>
    .small { font-size: 0.85rem; color: #888; }
    .prob { font-weight: 600; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------- Rutas esperadas -----------
RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "best_model.pth"
CLASSES_PATH = RESULTS_DIR / "classes.json"
SAMPLES_DIR = Path("data/samples")

# ----------- Transformaciones imagen -----------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_infer = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@st.cache_resource(show_spinner=False)
def load_classes() -> List[str]:
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"]
    return classes

@st.cache_resource(show_spinner=True)
def load_model(num_classes: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, device

def predict_image(model, device, img_pil: Image.Image, classes: List[str], topk=3):
    x = tf_infer(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    # top-k
    idxs = np.argsort(probs)[::-1][:topk]
    top = [(classes[i], float(probs[i])) for i in idxs]
    pred_class, pred_conf = top[0]
    return pred_class, pred_conf, top

def list_sample_images() -> List[Tuple[str, Path]]:
    """Devuelve pares (clase, ruta_imagen) leyendo data/samples/<clase>/*."""
    items = []
    if SAMPLES_DIR.exists():
        for cls_dir in sorted(SAMPLES_DIR.iterdir()):
            if cls_dir.is_dir():
                for img_file in sorted(cls_dir.iterdir()):
                    if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        items.append((cls_dir.name, img_file))
    return items

# ----------- Encabezado -----------
st.title("🪖 Clasificación de Vehículos Militares")
st.write("Transfer Learning con ResNet-50 (PyTorch)"
         "Sube una imagen o elige una muestra por categoría para ver la predicción.")

# ----------- Carga del modelo/clases -----------
if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("No encuentro `results/best_model.pth` o `results/classes.json`. "
             "Asegúrate de que existan en la carpeta `results/`.")
    st.stop()

classes = load_classes()
model, device = load_model(num_classes=len(classes))

# ----------- Barra lateral (controles) -----------
st.sidebar.header("Opciones")
topk = st.sidebar.slider("Top-K a mostrar", min_value=1, max_value=min(5, len(classes)), value=3, step=1)
conf_thresh = st.sidebar.slider("Umbral de confianza para resaltar", 0.0, 1.0, 0.50, 0.01)
show_probs = st.sidebar.checkbox("Mostrar tabla de probabilidades (Top-K)", value=True)

# Muestras por clase (si existen)
sample_items = list_sample_images()
by_class = {}
for cls, path in sample_items:
    by_class.setdefault(cls, []).append(path)

if by_class:
    st.sidebar.subheader("Muestras por categoría")
    picked_class = st.sidebar.selectbox("Clase", sorted(by_class.keys()))
    picked_img = st.sidebar.selectbox("Imagen de ejemplo", by_class[picked_class], format_func=lambda p: p.name)
    show_example = st.sidebar.button("Cargar ejemplo")
else:
    show_example = False

# ----------- Área principal: cargador o ejemplos -----------
col_u, col_ex = st.columns([2, 1], gap="large")

with col_u:
    st.subheader("Sube una imagen")
    files = st.file_uploader("Formatos: JPG/PNG. Puedes subir varias.", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


images_to_run = []

# Cargar ejemplos
if show_example and by_class:
    try:
        img = Image.open(picked_img).convert("RGB")
        images_to_run.append((picked_img.name, img))
    except Exception as e:
        st.warning(f"No pude abrir el ejemplo: {e}")

# Cargar subidas por el usuario
if files:
    for f in files:
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            images_to_run.append((f.name, img))
        except Exception as e:
            st.warning(f"No pude abrir {f.name}: {e}")

st.markdown("---")

# ----------- Predicción -----------
if not images_to_run:
    st.info("Sube una imagen o carga un ejemplo para ver la predicción.")
else:
    # grid responsivo
    cols_per_row = 2 if len(images_to_run) <= 2 else 3
    rows = (len(images_to_run) + cols_per_row - 1) // cols_per_row

    idx = 0
    for _ in range(rows):
        row_cols = st.columns(cols_per_row, gap="large")
        for c in row_cols:
            if idx >= len(images_to_run): break
            name, img = images_to_run[idx]
            idx += 1

            with c:
                st.image(img, caption=name, use_column_width=True)
                pred_class, pred_conf, top = predict_image(model, device, img, classes, topk=topk)
                color = "🟢" if pred_conf >= conf_thresh else "🟡"
                st.markdown(f"**Predicción:** {color} {pred_class} &nbsp; · &nbsp; "
                            f"**Confianza:** <span class='prob'>{pred_conf:.3f}</span>", unsafe_allow_html=True)

                if show_probs:
                    # pequeña tabla
                    st.write(
                        {cls: f"{p:.3f}" for cls, p in top}
                    )

st.markdown("---")
st.caption("Modelo: ResNet-50 fine-tuned | Normalización tipo ImageNet | Inference en CPU/GPU automática.")
