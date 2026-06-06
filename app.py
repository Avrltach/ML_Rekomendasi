import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Rekomendasi Divisi Pramuka", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }

    /* Judul halaman — hanya tag h1 di luar result-box */
    .page-title {
        color: #1E5128;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        padding-bottom: 10px;
        border-bottom: 3px solid #1E5128;
        margin-bottom: 20px;
    }
    .section-title { color: #1E5128; }

    /* Tombol */
    .stButton>button {
        background-color: #1E5128;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3E7C17;
    }

    /* Result box — gunakan class khusus, bukan override h1/h3 global */
    .result-box {
        background-color: #D8E9A8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E5128;
        margin-top: 20px;
    }
    .result-label {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .result-divisi {
        text-align: center;
        color: #1E5128;
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .result-desc {
        text-align: center;
        color: #333;
        font-size: 0.95rem;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Koneksi Google Sheets ────────────────────────────────────────
def init_connection():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
        if "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=scopes
            )
        else:
            creds = Credentials.from_service_account_file(
                "credentials.json", scopes=scopes
            )
        return gspread.authorize(creds)
    except FileNotFoundError:
        st.error("File credentials.json tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Gagal autentikasi Google: {e}")
        return None


# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

artifacts = load_model()
if artifacts is None:
    st.error("File model_artifacts.pkl tidak ditemukan. Jalankan notebook training terlebih dahulu.")
    st.stop()

# Validasi kunci artifacts
required_keys = ['model', 'encoders', 'fitur_kolom', 'target_col']
missing = [k for k in required_keys if k not in artifacts]
if missing:
    st.error(f"Artifacts tidak lengkap. Kunci yang hilang: {missing}")
    st.stop()

model      = artifacts['model']
encoders   = artifacts['encoders']
fitur_kolom = artifacts['fitur_kolom']
target_col = artifacts['target_col']


# ── Header ───────────────────────────────────────────────────────
st.markdown('<p class="page-title">Sistem Rekomendasi Divisi Pramuka</p>', unsafe_allow_html=True)
st.markdown("""
Selamat datang di sistem penentuan divisi berbasis **Machine Learning**.  
Sistem ini akan menganalisis minat dan bakat Anda untuk merekomendasikan divisi yang paling tepat.
""")

with st.expander("Petunjuk Pengisian"):
    st.markdown("""
    1. Isi **Nama Lengkap** dan **Kelas** dengan benar.
    2. Status otomatis terisi **Calon Dewan**.
    3. Jawab pertanyaan kuesioner pada skala 1–5.
    4. Tekan tombol **Proses Rekomendasi** di bawah.
    """)

st.markdown("---")


# ── Form ─────────────────────────────────────────────────────────
with st.form("form_rekomendasi"):
    col1, col2, col3 = st.columns(3)
    with col1:
        nama  = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap...")
    with col2:
        kelas = st.text_input("Kelas", placeholder="Contoh: X.1")
    with col3:
        st.text_input("Status", value="Calon Dewan", disabled=True)

    st.markdown('<p class="section-title"><strong>Kuesioner Minat & Bakat</strong></p>',
                unsafe_allow_html=True)
    st.caption("Skala 1 (Sangat Tidak Setuju) hingga 5 (Sangat Setuju)")

    col_kiri, col_kanan = st.columns(2)
    input_user = {}

    for i, col in enumerate(fitur_kolom):
        target_form = col_kiri if i % 2 == 0 else col_kanan
        with target_form:
            if col == 'Status':
                input_user[col] = "Calon Dewan"
            elif col in encoders:
                options = encoders[col].classes_.tolist()
                input_user[col] = st.selectbox(col, options)
            else:
                input_user[col] = st.slider(col, 1, 5, 3)

    st.markdown("")
    submitted = st.form_submit_button("PROSES REKOMENDASI")


# ── Prediksi ─────────────────────────────────────────────────────
if submitted:
    if not nama.strip() or not kelas.strip():
        st.warning("Nama dan Kelas wajib diisi.")
    else:
        # Buat dataframe input
        df_input = pd.DataFrame([input_user])[fitur_kolom]

        # Encode kolom kategorikal
        for col in df_input.columns:
            if col == 'Status':
                if col in encoders:
                    # Tampilkan pilihan sesuai data training, disabled
                    options = encoders[col].classes_.tolist()
                    input_user[col] = st.selectbox("Status", options, disabled=True)
                else:
                    input_user[col] = "Calon Dewan"

        # Prediksi
        pred         = model.predict(df_input)[0]
        hasil_divisi = encoders[target_col].inverse_transform([pred])[0]

        # Tampilkan hasil
        st.markdown("---")
        st.markdown(f"""
        <div class="result-box">
            <p class="result-label">Rekomendasi Divisi untuk Anda</p>
            <p class="result-divisi">{hasil_divisi}</p>
            <p class="result-desc">{nama} ({kelas}) direkomendasikan untuk bergabung di divisi ini.</p>
        </div>
        """, unsafe_allow_html=True)

        # Simpan ke Google Sheets
        try:
            client = init_connection()
            if client:
                SPREADSHEET_ID = '1DS2XgPwtqnCV7wOAumq02X4IdbkiZ5abS5df2x28D88'
                sheet    = client.open_by_key(SPREADSHEET_ID).sheet1
                row_data = [nama, kelas] + [input_user[col] for col in fitur_kolom] + [hasil_divisi]
                sheet.append_row(row_data)
                st.success("Data Anda berhasil tersimpan.")
        except gspread.exceptions.SpreadsheetNotFound:
            st.error("Spreadsheet tidak ditemukan. Pastikan SPREADSHEET_ID sudah benar dan sudah di-share ke service account.")
        except gspread.exceptions.APIError as e:
            st.error(f"Google Sheets API error: {e}")
        except Exception as e:
            st.error(f"Terjadi error saat menyimpan data: {e}")