import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

# =====================
# 1. Load Model
# =====================
MODEL_PATH = "cnn_deforestation_model.keras"   # ganti sesuai nama file model
model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# 2. Load Label Mapping
# =====================
# Biasanya dari train_generator.class_indices
# Contoh default (ubah sesuai dataset kamu)
class_indices = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9
}
# urutkan sesuai index
class_labels = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

# =====================
# 3. Streamlit UI
# =====================
st.title("🌍 Deforestation Classification App")
st.write("Upload citra satelit untuk memprediksi jenis lahan.")
# st.sidebar.title("⚙️ Pengaturan")
# st.sidebar.info("Upload gambar satelit untuk klasifikasi lahan.")
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        color: green;
        font-weight: bold;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 12px;
        color: gray;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="big-font">🌳 This model is using Basic Convolutional Neural Network</p>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # tampilkan gambar
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    # =====================
    # 4. Preprocessing
    # =====================
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0   # sama kayak di ImageDataGenerator(rescale=1./255)


    # =====================
    # 5. Prediksi
    # =====================
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_label = class_labels[pred_idx]
    pred_prob = preds[0][pred_idx]

    # =====================
    # 6. Output
    # =====================
    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Kelas Prediksi:** {pred_label}")
    st.write(f"**Probabilitas:** {pred_prob:.4f}")

    # tampilkan semua probabilitas
    st.bar_chart(preds[0])
    st.write({class_labels[i]: float(preds[0][i]) for i in range(len(class_labels))})