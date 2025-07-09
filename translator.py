import streamlit as st
import time
import base64
import os

# pip install streamlit-lottie
from streamlit_lottie import st_lottie
import requests

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="LaTeXTrans",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# ---------- 标题 ----------
st.markdown("""
<style>
.title-glow {
  font-size: 3em;
  font-weight: bold;
  text-align: center;
  background: linear-gradient(-45deg, #ff8a00, #e52e71, #9b00ff, #00eaff);
  background-size: 300% 300%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: rainbow 5s ease infinite, glow 2s ease-in-out infinite alternate;
}

@keyframes rainbow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes glow {
  from { text-shadow: 0 0 10px #ff8a00; }
  to   { text-shadow: 0 0 20px #e52e71; }
}
</style>

<h1 class='title-glow'>LaTeXTrans 🚀</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- 用户输入区域 ----------
arxiv_id = st.text_input("Please enter ArXiv ID:", placeholder="Such as: 2305.12345")

col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("Source Language", ["English", "Chinese", "Japanese", "Korean"], index=0)
with col2:
    target_lang = st.selectbox("Target Language", ["Chinese", "English", "Japanese", "Korean"], index=0)

# ---------- 模拟 PDF ----------
translated_pdf_path = "final_refine_ds_r1.pdf"
source_pdf_path = "DeepSeek_R1.pdf"

for path, text in [(translated_pdf_path, "This is a dummy translated PDF."),
                   (source_pdf_path, "This is a dummy original PDF.")]:
    if not os.path.exists(path):
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=text, ln=True, align='C')
            pdf.output(path)
        except:
            st.error("请安装 fpdf：pip install fpdf")

# ---------- 编码 PDF ----------
def encode_pdf_base64(pdf_path):
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------- Lottie 动画 URL 加载 ----------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return None

lottie_ai = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_nnpnmv0b.json")

# ---------- 翻译按钮 ----------
if st.button("🔁 Translate Now", use_container_width=True):
    if not arxiv_id:
        st.warning("⚠️ Please enter a valid ArXiv ID!")
    elif source_lang == target_lang:
        st.warning("⚠️ Source and target language cannot be the same.")
    else:
        if lottie_ai:
            st_lottie(lottie_ai, height=200, key="thinking")
        else:
            st.info("🤖 Translating...")

        st.info(f"Translating `{arxiv_id}` from {source_lang} to {target_lang}...")

        # 多阶段进度条
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.info("📥 Downloading paper from arXiv...")
        for i in range(30):
            time.sleep(0.2)
            progress_bar.progress(i + 1)

        status_text.info("🧠 Translating LaTeX content...")
        for i in range(30, 85):
            time.sleep(0.2)
            progress_bar.progress(i + 1)

        status_text.info("🧾 Generating translated PDF file...")
        for i in range(85, 100):
            time.sleep(0.2)
            progress_bar.progress(i + 1)

        progress_bar.empty()
        status_text.success("✅ Translation pipeline complete!")
        st.balloons()

        # ---------- 单栏预览 ----------
        st.markdown("### 📄 Preview Translation Result")
        translated_pdf_b64 = encode_pdf_base64(translated_pdf_path)

        pdf_display = f'''
        <div style="
            margin-top: 30px;
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 20px;
        ">
            <iframe
                src="data:application/pdf;base64,{translated_pdf_b64}#zoom=150"
                width="100%"
                height="1600px"
                style="border: none;"
                type="application/pdf">
            </iframe>
        </div>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)

        # ---------- 双栏同步预览 ----------
        st.markdown("### 📖 Original vs. Translated PDF Preview")
        source_pdf_b64 = encode_pdf_base64(source_pdf_path)

        dual_pdf_html = f'''
        <style>
        .pdf-container {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 30px;
            flex-wrap: wrap;
        }}
        .pdf-frame {{
            width: 48%;
            height: 800px;
            border: none;
            overflow: hidden;
            background: #fff;
            box-shadow: 0 0 12px rgba(0,0,0,0.2);
            border-radius: 12px;
        }}
        @media (max-width: 1000px) {{
            .pdf-frame {{ width: 100%; }}
        }}
        </style>

        <div class="pdf-container">
            <iframe class="pdf-frame" src="data:application/pdf;base64,{source_pdf_b64}#toolbar=0"></iframe>
            <iframe class="pdf-frame" src="data:application/pdf;base64,{translated_pdf_b64}#toolbar=0"></iframe>
        </div>
        '''
        st.markdown(dual_pdf_html, unsafe_allow_html=True)
