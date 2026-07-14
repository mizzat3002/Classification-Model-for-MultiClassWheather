import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

# ============================================================
# Konfigurasi
# ============================================================
WEIGHTS_PATH = 'model_tl_mobilenetv2.weights.h5'
IMG_HEIGHT = 220
IMG_WIDTH = 220

CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

CLASS_EMOJI = {
    'Cloudy': '☁️',
    'Rain': '🌧️',
    'Shine': '☀️',
    'Sunrise': '🌅'
}


@st.cache_resource
def load_model():
    # Bangun ulang arsitektur -- HARUS identik sama model_tl di notebook
    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.load_weights(WEIGHTS_PATH)   # <- load_weights, BUKAN load_model
    return model


def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run():
    st.title('🔮 Prediksi Kondisi Cuaca')
    st.markdown('''
    Unggah sebuah foto (langit/kondisi cuaca), lalu model akan menebak apakah foto
    tersebut termasuk kondisi **Cloudy**, **Rain**, **Shine**, atau **Sunrise**.
    ''')
    st.markdown('---')

    try:
        model = load_model()
    except Exception as e:
        st.error(f'Gagal memuat model. Pastikan file `{WEIGHTS_PATH}` ada satu folder '
                  f'dengan `prediction.py`. Detail error: {e}')
        return

    uploaded_file = st.file_uploader(
        'Upload gambar di sini (format JPG/JPEG/PNG)',
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption='Gambar yang diunggah', use_container_width=True)

        with col2:
            with st.spinner('Model sedang menebak...'):
                img_array = preprocess_image(image)
                pred_proba = model.predict(img_array, verbose=0)[0]
                pred_idx = int(np.argmax(pred_proba))
                pred_label = CLASS_NAMES[pred_idx]
                confidence = pred_proba[pred_idx] * 100

            st.success(f'### {CLASS_EMOJI[pred_label]} Prediksi: **{pred_label}**')
            st.metric('Tingkat Keyakinan Model', f'{confidence:.2f}%')

            df_proba = pd.DataFrame({
                'Kelas': CLASS_NAMES,
                'Probabilitas (%)': (pred_proba * 100).round(2)
            }).sort_values('Probabilitas (%)', ascending=False).reset_index(drop=True)

            st.dataframe(df_proba, use_container_width=True, hide_index=True)
            st.bar_chart(df_proba.set_index('Kelas'))

        st.markdown('---')
        st.caption('''
        ⚠️ **Catatan:** Model ini dilatih hanya pada 4 kategori cuaca (Cloudy, Rain,
        Shine, Sunrise) dengan dataset yang terbatas.
        ''')
    else:
        st.info('Silakan unggah gambar terlebih dahulu untuk melihat hasil prediksi.')
