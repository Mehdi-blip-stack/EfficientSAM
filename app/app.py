"""EfficientSAM demo: promptable segmentation with our trained model (Streamlit, written by our team).

Run from the repository root:   streamlit run app/app.py
"""
import hashlib
import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from segmenter import Segmenter, available_models

HERE = Path(__file__).absolute().parent
EXAMPLES = {"Cat, toilet and sink": "000000157807.jpg", "Zebras": "000000215245.jpg",
            "Sheep": "000000397639.jpg", "Buses": "000000531036.jpg"}
DISPLAY_W = 800                      # resolution of the clickable image (it is scaled to the column width)
COLORS = [(30, 144, 255), (255, 140, 0)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="EfficientSAM demo", layout="wide")


# ------------------------------------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model ...")
def load_model(name):
    kind, weights = available_models()[name]
    return Segmenter(name, kind, weights, DEVICE)


def get_embedding(model, img, img_key):
    """The image encoder runs once per (model, image); later prompts reuse the embedding."""
    cache = st.session_state.setdefault("emb", {})
    k = (model.name, img_key)
    if k not in cache:
        emb, ms = model.encode(img)
        cache[k] = (emb, ms)
        return emb, ms, False
    return cache[k][0], cache[k][1], True


def overlay(img, mask, color, alpha=0.5):
    out = img.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, np.float32)
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    out[edge] = color
    return out.astype(np.uint8)


def draw_prompt(pil, prompt, scale=1.0, width=3):
    pil = pil.copy(); d = ImageDraw.Draw(pil)
    if prompt is None:
        return pil
    if prompt["type"] == "box":
        x0, y0, x1, y1 = [v * scale for v in prompt["xyxy"]]
        d.rectangle([x0, y0, x1, y1], outline=(0, 230, 118), width=width)
    else:
        x, y = [v * scale for v in prompt["xy"]]; r = 7
        d.ellipse([x - r, y - r, x + r, y + r], fill=(0, 230, 118), outline=(255, 255, 255), width=2)
    return pil


def png_bytes(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()


# ------------------------------------------------------------------------------------------------ sidebar
models = available_models()
st.sidebar.title("Settings")
if not models:
    st.error("No model weights found. Put our fine-tuned checkpoints (*.pt) in app/weights/ "
             "and keep the released EfficientSAM weights in weights/ (see app/README.md).")
    st.stop()
names = list(models)
model_a = st.sidebar.selectbox("Model", names, index=0)
compare = st.sidebar.checkbox("Compare with a second model", value=False)
model_b = None
if compare and len(names) > 1:
    default_b = next((i for i, n in enumerate(names) if "released" in n and n != model_a), 0)
    model_b = st.sidebar.selectbox("Second model", [n for n in names], index=default_b)
show_all = st.sidebar.checkbox("Show all 3 candidate masks", value=False)
st.sidebar.caption(f"Running on: **{DEVICE.upper()}**")

with st.sidebar.expander("About the models"):
    st.markdown(
        "**Ours**: EfficientSAM-Ti architecture (10.2 M parameters) at 256×256 px, encoder pretrained by us with "
        "SAMI (SAM ViT-H as teacher) on 800 COCO images, then fine-tuned for box and click prompts on 800 images "
        "(`SAMI_finetuned`) or 5,000 images (`SAMI_finetuned_5k`, our best), on Colab T4 GPUs.\n\n"
        "**Released**: the authors' EfficientSAM-Ti/S, trained on ImageNet-1K and SA-1B (11 M images) at "
        "1024×1024 px on 64 A100 GPUs.\n\n"
        "On our 1,620 held-out COCO objects (mIoU, box / click): released EfficientSAM-Ti 0.751 / 0.501, "
        "-S 0.764 / 0.598; ours (SAMI start) 0.577 / 0.313 with 5,000 fine-tuning images, 0.497 / 0.268 "
        "with 800, and 0.487 / 0.257 with the encoder frozen. Expect coarser masks from our model, "
        "especially on small objects.")

# ------------------------------------------------------------------------------------------------ input image
st.title("EfficientSAM: promptable segmentation demo")
st.caption("**Click** on an object for a point prompt, or **click and drag** a rectangle for a box prompt. "
           "The model returns 3 candidate masks and shows the one with the highest predicted IoU.")

src = st.radio("Image", ["Example", "Upload your own"], horizontal=True, label_visibility="collapsed")
if src == "Example":
    choice = st.selectbox("Example image (COCO val2017, not used in training)", list(EXAMPLES))
    raw = (HERE / "examples" / EXAMPLES[choice]).read_bytes()
else:
    up = st.file_uploader("JPEG or PNG image", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Upload an image to start.")
        st.stop()
    raw = up.getvalue()

img_key = hashlib.md5(raw).hexdigest()
if st.session_state.get("img_key") != img_key:              # new image: forget the old prompt and embeddings
    st.session_state.update(img_key=img_key, prompt=None, last_event=None, emb={})
pil = Image.open(io.BytesIO(raw)).convert("RGB")             # always 3-channel RGB (greyscale/RGBA safe)
img = np.array(pil)
H, W = img.shape[:2]

# ------------------------------------------------------------------------------------------------ layout
col_in, col_a = st.columns(2)
with col_in:
    st.subheader("Input")
    disp_w = min(DISPLAY_W, W)
    scale = disp_w / W
    shown = draw_prompt(pil.resize((disp_w, round(H * scale))), st.session_state.prompt, scale=scale)
    ev = streamlit_image_coordinates(shown, key="canvas", click_and_drag=True, cursor="crosshair", width="stretch")
    if ev is not None and ev.get("unix_time") != st.session_state.last_event:
        st.session_state.last_event = ev.get("unix_time")
        f = W / ev["width"]                                  # on-screen pixels -> original pixels
        x1, y1, x2, y2 = [min(max(v, 0), lim) * f for v, lim in
                          [(ev["x1"], ev["width"]), (ev["y1"], ev["height"]), (ev["x2"], ev["width"]), (ev["y2"], ev["height"])]]
        if abs(x2 - x1) < 5 * f and abs(y2 - y1) < 5 * f:      # a click, not a drag
            st.session_state.prompt = {"type": "point", "xy": (x1, y1)}
        else:
            st.session_state.prompt = {"type": "box", "xyxy": (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))}
        st.rerun()
    if st.button("Clear prompt", disabled=st.session_state.prompt is None):
        st.session_state.prompt = None; st.rerun()
    st.caption(f"Image size: {W} × {H} px")

prompt = st.session_state.prompt


def run(name, col, color):
    with col:
        st.subheader(name)
        model = load_model(name)
        if model.info:
            st.caption(" · ".join(f"{k}: {v}" for k, v in model.info.items()))
        if prompt is None:
            st.info("Waiting for a prompt: click or drag on the input image.")
            return None
        with st.spinner("Encoding image ..." if (name, img_key) not in st.session_state.emb else "Predicting ..."):
            emb, enc_ms, cached = get_embedding(model, img, img_key)
            pred = model.predict(emb, (H, W), point=prompt.get("xy"), box=prompt.get("xyxy"))
        mask = pred.masks[pred.best]
        out = np.array(draw_prompt(Image.fromarray(overlay(img, mask, color)), prompt))
        st.image(out, width="stretch")
        c1, c2 = st.columns(2)
        c1.metric("Encoder (once per image)", f"{enc_ms:,.0f} ms", "cached" if cached else "this image",
                  delta_color="off")
        c2.metric("Decoder (this prompt)", f"{pred.decoder_ms:,.0f} ms")
        c3, c4 = st.columns(2)
        c3.metric("Predicted IoU", f"{pred.iou[pred.best]:.2f}")
        c4.metric("Mask area (of image)", f"{100 * mask.mean():.1f}%")
        if show_all:
            cols = st.columns(3)
            for i, c in enumerate(cols):
                c.image(overlay(img, pred.masks[i], color), width="stretch",
                        caption=f"Mask {i + 1}: predicted IoU {pred.iou[i]:.2f}" + (" (shown)" if i == pred.best else ""))
        d1, d2 = st.columns(2)
        tag = name.split(":")[-1].strip().replace(" ", "_").replace("(", "").replace(")", "")
        d1.download_button("Download mask (PNG)", png_bytes(mask.astype(np.uint8) * 255),
                           file_name=f"mask_{tag}.png", mime="image/png", key=f"dm_{name}", on_click="ignore")
        d2.download_button("Download overlay (PNG)", png_bytes(out), file_name=f"overlay_{tag}.png",
                           mime="image/png", key=f"do_{name}", on_click="ignore")
        return {"Model": name, "Parameters (M)": f"{model.params_m:.1f}", "Encoder (ms)": f"{enc_ms:,.0f}",
                "Decoder (ms)": f"{pred.decoder_ms:,.0f}", "Predicted IoU": f"{pred.iou[pred.best]:.2f}",
                "Mask area (%)": f"{100 * mask.mean():.1f}"}, mask


res_a = run(model_a, col_a, COLORS[0])                     # model A: blue
if compare and model_b:
    col_b, col_cmp = st.columns(2)
    res_b = run(model_b, col_b, COLORS[1])                 # model B: orange
    with col_cmp:
        st.subheader("Comparison")
        if res_a and res_b:
            (ra, ma), (rb, mb) = res_a, res_b
            agree = (ma & mb).sum() / max((ma | mb).sum(), 1)
            keys = [k for k in ra if k != "Model"]
            st.table({"": keys, "A: " + ra["Model"]: [ra[k] for k in keys], "B: " + rb["Model"]: [rb[k] for k in keys]})
            st.metric("Agreement between the two masks (IoU)", f"{agree:.2f}")
            st.caption("Encoder times include the first run only; later prompts on the same image reuse the "
                       "cached embedding, so only the decoder runs.")
