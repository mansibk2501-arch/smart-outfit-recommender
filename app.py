import colorsys
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans

from color_utils import (
    COLOR_PALETTE, hex_to_rgb, rgb_to_hex, rgb_to_hsv,
    color_features, harmony_rule_score, nearest_palette_name
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "outfit_color_model.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"

st.set_page_config(page_title="Smart Outfit Color Recommender", page_icon="👗", layout="wide")

st.title("👗 Smart Outfit Color Recommendation System")
st.caption("Image color detection • Color harmony • ML-based ranking")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

with st.sidebar:
    st.header("How it works")
    st.write("1. Choose a color or upload a clothing photo.")
    st.write("2. The app detects/analyzes the main color.")
    st.write("3. Color-harmony rules and a trained ML classifier rank suggestions.")
    st.caption("The starter model is trained on synthetic labels generated from color-harmony rules. Its metrics measure agreement with those rules, not human fashion preference.")

tab1, tab2, tab3 = st.tabs(["🎨 Choose a color", "📷 Upload a photo", "📊 Model evaluation"])

def recommend(base_rgb, occasion, n=6):
    rows = []
    base_hex = rgb_to_hex(base_rgb)
    for name, hex_value in COLOR_PALETTE.items():
        candidate_rgb = hex_to_rgb(hex_value)
        if np.allclose(base_rgb, candidate_rgb, atol=8):
            continue
        feats = color_features(base_rgb, candidate_rgb)
        rule_score = harmony_rule_score(base_rgb, candidate_rgb)
        ml_score = float(model.predict_proba(np.array(feats).reshape(1, -1))[0][1]) if model is not None else rule_score
        occasion_bonus = 0.0
        occasion_favorites = {
            "College / Casual": ["White", "Grey", "Beige", "Navy Blue", "Black"],
            "Party": ["Black", "Red", "Gold", "White", "Purple"],
            "Wedding / Festive": ["Gold", "Maroon", "Green", "Beige", "Pink"],
            "Minimal / Neutral": ["White", "Black", "Grey", "Beige", "Brown"],
        }
        if name in occasion_favorites[occasion]:
            occasion_bonus = 0.06
        final_score = 0.70 * ml_score + 0.30 * rule_score + occasion_bonus
        rows.append({
            "Color": name,
            "HEX": hex_value,
            "ML score": round(ml_score, 3),
            "Harmony score": round(rule_score, 3),
            "Final score": round(min(final_score, 1.0), 3),
        })
    rows.sort(key=lambda x: x["Final score"], reverse=True)
    return rows[:n]

def show_recommendations(base_rgb, occasion, item_label):
    base_hex = rgb_to_hex(base_rgb)
    name = nearest_palette_name(base_rgb)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("**Detected / selected color**")
        st.markdown(
            f'<div style="background:{base_hex};height:100px;border-radius:12px;border:1px solid #999"></div>',
            unsafe_allow_html=True
        )
        st.write(f"**{name}**")
        st.code(base_hex)
    with c2:
        st.markdown(f"**Matching {item_label.lower()} for: {occasion}**")
        recs = recommend(base_rgb, occasion)
        cols = st.columns(3)
        for i, rec in enumerate(recs):
            with cols[i % 3]:
                st.markdown(
                    f'<div style="background:{rec["HEX"]};height:76px;border-radius:10px;border:1px solid #999"></div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{rec['Color']}**")
                st.caption(f'{rec["HEX"]} · score {rec["Final score"]:.2f}')
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)

with tab1:
    st.subheader("Choose a clothing color")
    names = list(COLOR_PALETTE.keys())
    selected = st.selectbox("Base clothing color", names, index=names.index("Pink"))
    occasion = st.selectbox("Occasion / style", ["College / Casual", "Party", "Wedding / Festive", "Minimal / Neutral"], key="occasion_picker")
    item = st.selectbox("Recommend colors for", ["Trousers / Jeans", "Dupatta", "Shirt / Top", "Shoes / Accessories"])
    if st.button("✨ Recommend colors", key="recommend_palette", type="primary"):
        show_recommendations(hex_to_rgb(COLOR_PALETTE[selected]), occasion, item)

with tab2:
    st.subheader("Upload a clothing photo")
    st.write("Tip: use a clear, well-lit photo with one main garment. Busy backgrounds and shadows can affect detection.")
    uploaded = st.file_uploader("Upload JPG, JPEG or PNG", type=["jpg", "jpeg", "png"])
    occasion2 = st.selectbox("Occasion / style", ["College / Casual", "Party", "Wedding / Festive", "Minimal / Neutral"], key="occasion_image")
    item2 = st.selectbox("Recommend colors for", ["Trousers / Jeans", "Dupatta", "Shirt / Top", "Shoes / Accessories"], key="item_image")
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded photo", use_container_width=True)
        # Resize before clustering to make processing fast
        image.thumbnail((240, 240))
        pixels = np.asarray(image).reshape(-1, 3)
        # Exclude near-white/near-black extremes only lightly; keep real clothing colors too.
        if len(pixels) > 5000:
            rng = np.random.default_rng(42)
            pixels = pixels[rng.choice(len(pixels), 5000, replace=False)]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        counts = np.bincount(labels)
        dominant_rgb = np.rint(kmeans.cluster_centers_[int(np.argmax(counts))]).astype(int)
        dominant_hex = rgb_to_hex(dominant_rgb)
        st.markdown("**Detected dominant image color**")
        st.markdown(
            f'<div style="background:{dominant_hex};height:70px;width:160px;border-radius:10px;border:1px solid #999"></div>',
            unsafe_allow_html=True
        )
        st.code(dominant_hex)
        st.caption("Prototype limitation: this finds a dominant color in the whole image, not specifically the garment. A plain background gives better results.")
        if st.button("Recommend from detected color", type="primary"):
            show_recommendations(dominant_rgb, occasion2, item2)

with tab3:
    st.subheader("Model evaluation")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("Test accuracy", f'{metrics.get("accuracy", 0):.1%}')
        b.metric("Test precision", f'{metrics.get("precision", 0):.1%}')
        c.metric("Test recall", f'{metrics.get("recall", 0):.1%}')
        st.write(f"Training rows: **{metrics.get('train_rows', 'N/A')}** · Test rows: **{metrics.get('test_rows', 'N/A')}**")
        st.info(metrics.get("note", "Metrics are calculated on held-out synthetic examples."))
        st.write("Confusion matrix (true labels by rows, predicted labels by columns):")
        st.write(pd.DataFrame(metrics.get("confusion_matrix", []), index=["Rule: not-match", "Rule: match"], columns=["Predicted not-match", "Predicted match"]))
        st.write("Classification report:")
        st.json(metrics.get("classification_report", {}))
    else:
        st.warning("Metrics file not found yet. Run `python train_model.py` in the project terminal, then refresh this page.")

st.divider()
st.caption("Educational prototype. Color recommendations are suggestions, not universal fashion rules.")
