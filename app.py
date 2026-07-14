import streamlit as st
import eda
import prediction

# ============================================================
# Konfigurasi halaman utama
# ============================================================
st.set_page_config(
    page_title='Weather Image Classification',
    page_icon='⛅',
    layout='wide'
)


def main():
    # ----- Sidebar navigasi -----
    st.sidebar.title('🧭 Navigasi')
    page = st.sidebar.selectbox(
        'Pilih Halaman',
        ('Home', 'EDA', 'Prediksi')
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Graded Challenge 7**')
    st.sidebar.markdown('Computer Vision - Klasifikasi Cuaca')
    st.sidebar.markdown('by Muhammad Izzat')

    # ----- Routing antar halaman -----
    if page == 'Home':
        show_home()
    elif page == 'EDA':
        eda.run()
    elif page == 'Prediksi':
        prediction.run()


def show_home():
    st.title('⛅ Weather Image Classification App')
    st.markdown('---')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('''
        ### Selamat Datang! 👋

        Aplikasi ini dibuat untuk mengklasifikasikan **kondisi cuaca** dari sebuah foto
        ke dalam salah satu dari **4 kategori**:

        - ☁️ **Cloudy** (Berawan)
        - 🌧️ **Rain** (Hujan)
        - ☀️ **Shine** (Cerah)
        - 🌅 **Sunrise** (Matahari Terbit/Terbenam)

        Model yang dipakai adalah hasil **Transfer Learning** menggunakan arsitektur
        **MobileNetV2** (pre-trained di ImageNet), yang kemudian dilatih ulang khusus
        untuk mengenali pola visual dari keempat kondisi cuaca di atas.

        ### Cara Pakai
        1. Buka halaman **EDA** di sidebar kalau kamu mau lihat eksplorasi data yang dipakai untuk melatih model.
        2. Buka halaman **Prediksi** untuk mengunggah foto kamu sendiri dan lihat hasil klasifikasinya.

        ### Tujuan Aplikasi
        Aplikasi ini ditujukan sebagai modul pembantu otomatisasi pelabelan cuaca dari foto
        (misalnya untuk aplikasi logistik/ride-hailing atau platform berbagi foto),
        **bukan** sebagai pengganti prakiraan cuaca resmi dari BMKG atau lembaga meteorologi lainnya.
        ''')

    with col2:
        st.info('''
        **Model Info**
        - Arsitektur: MobileNetV2 (Transfer Learning)
        - Input size: 220 x 220 px
        - Jumlah kelas: 4
        - Akurasi test-set: ±94%
        ''')


if __name__ == '__main__':
    main()
