import streamlit as st
import eda
import prediction


st.set_page_config(
    page_title='Weather Image Classification',
    layout='wide'
)


def main():
    page = st.sidebar.selectbox(
        'Pilih Halaman',
        ('Home', 'EDA', 'Prediksi')
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Graded Challenge 7**')
    st.sidebar.markdown('Computer Vision - Klasifikasi Cuaca')
    st.sidebar.markdown('by Muhammad Izzat')

    if page == 'Home':
        show_home()
    elif page == 'EDA':
        eda.run()
    elif page == 'Prediksi':
        prediction.run()


def show_home():
    st.title('Weather Image Classification App')
    st.image('gambar.jpg')
    st.markdown('---')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('''
        ### Welcome! 

        Aplikasi ini dibuat untuk mengklasifikasikan **kondisi cuaca** dari sebuah foto
        ke dalam salah satu dari **4 kategori**:

        -  **Cloudy** 
        -  **Rain** 
        -  **Shine** 
        -  **Sunrise** 

        Model yang dipakai adalah hasil **Transfer Learning** menggunakan arsitektur
        **MobileNetV2**, yang kemudian dilatih untuk mengenali pola visual dari keempat kondisi cuaca di atas.

        ### Tutorial 
        1. Buka halaman **EDA** di sidebar untuk lihat eksplorasi data yang dipakai untuk melatih model.
        2. Buka halaman **Prediksi** untuk mengupload image dan lihat hasil klasifikasinya.''')

    with col2:
        st.markdown('''
        **Model Info**
        - Arsitektur: MobileNetV2 (Transfer Learning)
        - Input size: 220 x 220 px
        - Jumlah kelas: 4
        - Akurasi test-set: ±94%
        ''')


if __name__ == '__main__':
    main()
