import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# ============================================================
# Konfigurasi
# ============================================================
MODEL_PATH = 'model_tl_mobilenetv2.keras'
IMG_HEIGHT = 220
IMG_WIDTH = 220

# Urutan kelas HARUS sama persis dengan class_indices dari train_set_aug
# sewaktu training (flow_from_directory mengurutkan nama folder secara alfabetis)
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

CLASS_EMOJI = {
    'Cloudy': '☁️',
    'Rain': '🌧️',
    'Shine': '☀️',
    'Sunrise': '🌅'
}


# ============================================================
# Load model (di-cache supaya tidak perlu load ulang setiap interaksi)
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image):
    '''
    Preprocessing gambar supaya konsisten dengan proses training:
    - resize ke 220x220
    - rescale nilai piksel dari 0-255 menjadi 0-1
    '''
    image = image.convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # tambah dimensi batch
    return img_array


def run():
    st.title('🔮 Prediksi Kondisi Cuaca')
    st.markdown('''
    Unggah sebuah foto (langit/kondisi cuaca), lalu model akan menebak apakah foto
    tersebut termasuk kondisi **Cloudy**, **Rain**, **Shine**, atau **Sunrise**.
    ''')
    st.markdown('---')

    # Load model sekali saja (cached)
    try:
        model = load_model()
    except Exception as e:
        st.error(f'Gagal memuat model. Pastikan file `{MODEL_PATH}` ada satu folder '
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

            st.markdown('**Detail probabilitas tiap kelas:**')
            df_proba = pd.DataFrame({
                'Kelas': CLASS_NAMES,
                'Probabilitas (%)': (pred_proba * 100).round(2)
            }).sort_values('Probabilitas (%)', ascending=False).reset_index(drop=True)

            st.dataframe(df_proba, use_container_width=True, hide_index=True)
            st.bar_chart(df_proba.set_index('Kelas'))

        st.markdown('---')
        st.caption('''
        ⚠️ **Catatan:** Model ini dilatih hanya pada 4 kategori cuaca (Cloudy, Rain,
        Shine, Sunrise) dengan dataset yang terbatas. Hasil prediksi bisa kurang akurat
        untuk foto dengan kondisi yang jauh berbeda dari data training (misalnya foto
        malam hari, salju, atau kabut tebal).
        ''')
    else:
        st.info('Silakan unggah gambar terlebih dahulu untuk melihat hasil prediksi.')
