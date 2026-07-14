import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model


WEIGHTS_PATH = 'model_tl_mobilenetv2.weights.h5'
IMG_HEIGHT = 220
IMG_WIDTH = 220
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Load Model 
@st.cache_resource
def load_my_model():
    # arsitektur model
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
    model.load_weights(WEIGHTS_PATH)   
    return model

# fungsi buat run app
def run():
    try:
        model = load_my_model()
    except Exception as e:
        st.error(f'Gagal memuat model. Pastikan file {WEIGHTS_PATH} ada di repo. Detail: {e}')
        return

# interface web
st.title('Prediksi Kondisi Cuaca')
st.write('Upload Image, program akan memprediksi kondisi cuaca')

# tombol untuk upload 
uploaded_file = st.file_uploader('Pilih file image (Format: JPG/PNG)', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # tampilkan gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    
    # preprocessing gambar 
    img_rgb = image.convert('RGB')
    img_resized = img_rgb.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # prrediksi 
    predictions = model.predict(img_array)[0]
    pred_idx = np.argmax(predictions)
    pred_label = CLASS_NAMES[pred_idx]
    confidence = predictions[pred_idx] * 100
    
    # hasil tebakan 
    st.write('---')
    st.write(f'### Hasil Tebakan: **{pred_label}**')
    st.write(f'Tingkat Keyakinan: **{confidence:.2f}%**')
    st.write('---')
