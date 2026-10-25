```python
import os
import hashlib
import html
import platform
import streamlit as st

from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================

st.set_page_config(
    page_title="De PDF a Cuento",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ESTILOS: MUNDO INFANTIL, ROSADO Y PASTEL
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

    :root {
        --rosa: #F28BB8;
        --rosa-fuerte: #D95791;
        --rosa-claro: #FFF0F6;
        --rosa-suave: #FFE0ED;
        --blanco: #FFFFFF;
        --lila: #E9DFFF;
        --amarillo: #FFF0B8;
        --azul: #DDF4FF;
        --texto: #593B50;
        --texto-suave: #866A7D;
    }

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 5% 5%, #FFE0ED 0, transparent 23%),
            radial-gradient(circle at 95% 20%, #E9DFFF 0, transparent 20%),
            #FFFAFC;
        color: var(--texto);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    h1, h2, h3 {
        color: #B9487D !important;
        font-weight: 900 !important;
    }

    p, label, .stMarkdown {
        color: var(--texto);
    }

    .hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF0F6 55%, #F7EEFF 100%);
        border: 2px solid #F9D3E4;
        border-radius: 32px;
        padding: 38px 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(217, 87, 145, 0.09);
        margin-bottom: 25px;
    }

    .hero-decoration {
        font-size: 30px;
        letter-spacing: 8px;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        color: #D95791 !important;
        margin: 0;
        line-height: 1.15;
    }

    .hero p {
        font-size: 1.1rem;
        color: #866A7D;
        max-width: 650px;
        margin: 15px auto 0 auto;
        line-height: 1.8;
    }

    .hero-tag {
        display: inline-block;
        background: #FFE0ED;
        color: #B9487D;
        border-radius: 30px;
        padding: 8px 18px;
        margin-top: 20px;
        font-weight: 800;
        font-size: 0.9rem;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.92);
        border: 2px solid #F8D9E7;
        border-radius: 24px;
        padding: 24px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(150, 80, 120, 0.05);
    }

    .section-title {
        font-size: 1.35rem;
        color: #B9487D;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .section-description {
        color: #866A7D;
        line-height: 1.7;
        font-size: 0.98rem;
    }

    .tip {
        background: #FFF0B8;
        border: 2px solid #FFE49A;
        padding: 15px 18px;
        border-radius: 18px;
        color: #765A36;
        line-height: 1.7;
        margin: 15px 0;
    }

    .story {
        background: #FFFFFF;
        border: 3px solid #F5C7DC;
        border-radius: 28px;
        padding: clamp(22px, 5vw, 45px);
        box-shadow: 0 12px 35px rgba(217, 87, 145, 0.09);
        margin-top: 18px;
    }

    .story-label {
        display: inline-block;
        background: #FFE0ED;
        color: #B9487D;
        padding: 7px 15px;
        border-radius: 30px;
        font-weight: 900;
        margin-bottom: 18px;
    }

    .story-body {
        font-size: 1.08rem;
        line-height: 2;
        color: #593B50;
        white-space: pre-wrap;
    }

    .stButton > button {
        background: linear-gradient(135deg, #F28BB8, #D95791);
        color: white;
        border: none;
        border-radius: 18px;
        padding: 0.7rem 1.3rem;
        font-family: 'Nunito', sans-serif;
        font-size: 1rem;
        font-weight: 900;
        min-height: 48px;
        box-shadow: 0 5px 12px rgba(217, 87, 145, 0.16);
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #E96CA5, #C84483);
        color: white;
        transform: translateY(-2px);
        border: none;
    }

    [data-testid="stFileUploader"] {
        background: #FFF5F9;
        border: 2px dashed #E9A5C4;
        border-radius: 20px;
        padding: 15px;
    }

    [data-testid="
