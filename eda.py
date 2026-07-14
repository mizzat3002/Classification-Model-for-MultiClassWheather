import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def run():
    st.title('📊 Exploratory Data Analysis (EDA)')
    st.markdown('''
    Halaman ini menampilkan ringkasan hasil eksplorasi data (EDA) yang dilakukan
    terhadap **Multi-class Weather Dataset** (1.125 gambar, 4 kelas cuaca) sebelum
    dipakai untuk melatih model.
    ''')
    st.markdown('---')

    # ============================================================
    # 1. Distribusi jumlah data per kelas
    # ============================================================
    st.subheader('1. Distribusi Data per Kelas')

    # Angka ini diambil dari hasil EDA di notebook (grand total = 1.125 gambar)
    distribusi = {
        'Cloudy': 300,
        'Rain': 215,
        'Shine': 253,
        'Sunrise': 357
    }
    df_distribusi = pd.DataFrame({
        'Kelas': list(distribusi.keys()),
        'Jumlah Gambar': list(distribusi.values())
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=df_distribusi, x='Kelas', y='Jumlah Gambar', palette='Set2', ax=ax)
        ax.set_title('Distribusi Data per Kelas Cuaca', fontsize=13, fontweight='bold')
        ax.set_xlabel('Kelas')
        ax.set_ylabel('Jumlah Gambar')
        for i, v in enumerate(df_distribusi['Jumlah Gambar']):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.dataframe(df_distribusi, use_container_width=True)
        st.metric('Total Gambar', '1.125')

    st.markdown('''
    **Insight:** Dataset ini punya distribusi yang lumayan merata di 4 kelas.
    Kelas `Sunrise` merupakan kelas mayoritas (357 gambar) dan kelas `Rain` sebagai
    kelas minoritas (215 gambar), tapi rasio perbandingan kelas mayoritas dan minoritas
    cuma sekitar 1.6:1 — tidak sampai 2x lipat, jadi dataset ini tergolong cukup seimbang
    dan tidak perlu penanganan khusus untuk *class imbalance*.
    ''')

    st.markdown('---')

    # ============================================================
    # 2. Ukuran dimensi gambar
    # ============================================================
    st.subheader('2. Ukuran Dimensi Gambar')
    st.markdown('''
    Gambar-gambar dalam dataset ini punya ukuran (dimensi piksel) yang **sangat
    beragam** — ada yang kecil, ada yang besar, dengan rasio lebar-tinggi yang
    berbeda-beda juga, karena diambil dari sumber foto yang berbeda-beda.

    Karena itu, semua gambar di-*resize* menjadi ukuran seragam **220 x 220 piksel**
    sebelum dipakai untuk melatih model.
    ''')

    st.markdown('---')

    # ============================================================
    # 3. Tingkat kecerahan (brightness) per kelas
    # ============================================================
    st.subheader('3. Tingkat Kecerahan (Brightness) per Kelas')

    brightness = {
        'Cloudy': 112.31,
        'Rain': 118.30,
        'Shine': 144.21,
        'Sunrise': 101.00
    }
    df_brightness = pd.DataFrame({
        'Kelas': list(brightness.keys()),
        'Rata-rata Brightness': list(brightness.values())
    })

    col1, col2 = st.columns([2, 1])
    with col1:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=df_brightness, x='Kelas', y='Rata-rata Brightness', palette='mako', ax=ax2)
        ax2.set_title('Rata-rata Tingkat Kecerahan per Kelas Cuaca', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Kelas')
        ax2.set_ylabel('Rata-rata Nilai Piksel (0-255)')
        st.pyplot(fig2)
    with col2:
        st.dataframe(df_brightness, use_container_width=True)

    st.markdown('''
    **Insight:**
    - Kelas `Shine` merupakan kelas paling cerah, dengan tingkat brightness sekitar **144**.
    - Kelas `Cloudy` dan `Rain` berada di tengah-tengah antara paling terang dan paling
      gelap, karena karakter dari kedua kelas ini memang agak redup — ini juga yang
      berpotensi membuat model tertukar mengklasifikasikan kedua kelas tersebut.
    - Kelas paling gelap dari dataset ini adalah `Sunrise`, karena gambarnya banyak
      didominasi warna hitam/gelap (siluet saat matahari terbit/terbenam).
    ''')

    st.markdown('---')
    st.caption('Sumber dataset: [Multi-class Weather Dataset - Kaggle]'
               '(https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)')
