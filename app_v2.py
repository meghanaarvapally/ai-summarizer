# -------------------------
# app_v2.py
# AI Text Summarizer v2 (Final)
# Author: Meghana Arvapally
# -------------------------

import streamlit as st
from transformers import pipeline
import pdfplumber
import docx
import pytesseract
from PIL import Image
import nltk
import time

# --- Download necessary NLTK data ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Tesseract Path for Windows ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Load Summarizer Model (BART) ---
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()


# --- Helper Functions ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()


def extract_text_from_images(images):
    text = ""
    progress = st.progress(0)
    total = len(images)
    for idx, img in enumerate(images):
        image = Image.open(img)
        text += pytesseract.image_to_string(image, config="--psm 6") + "\n"
        progress.progress((idx + 1) / total)
        time.sleep(0.2)
    progress.empty()
    return text.strip()


def summarize_large_text(text, summary_length="Medium"):
    # Split long text into manageable chunks
    words = text.split()
    summaries = []
    chunk_size = 800

    length_settings = {
        "Short": {"max": 80, "min": 30},
        "Medium": {"max": 150, "min": 60},
        "Detailed": {"max": 250, "min": 100}
    }
    max_len = length_settings[summary_length]["max"]
    min_len = length_settings[summary_length]["min"]

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            summaries.append(summary[0]['summary_text'])

    combined_text = " ".join(summaries)

    # Optional final compression
    if len(combined_text.split()) > 500:
        final_summary = summarizer(combined_text, max_length=max_len, min_length=min_len, do_sample=False)
        return final_summary[0]['summary_text']

    return combined_text


def is_resume(text):
    keywords = ["experience", "education", "skills", "projects", "certifications", "achievements"]
    match_count = sum(1 for k in keywords if k.lower() in text.lower())
    return match_count >= 3 and len(text.split()) < 1200


# --- Streamlit UI ---
st.set_page_config(page_title="AI Text Summarizer v2", layout="wide")

st.title("🧠 AI Text Summarizer")
st.write("Summarize PDFs, Word Docs, Text Files, or multiple Images — or paste text manually!")

# File uploaders separated by type
col1, col2 = st.columns(2)

with col1:
    uploaded_docs = st.file_uploader(
        "📄 Upload Document Files (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

with col2:
    uploaded_images = st.file_uploader(
        "🖼️ Upload up to 5 Images (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

manual_text = st.text_area("📝 Or paste text manually:", height=200)
summary_length = st.selectbox("Select summary length (for normal docs only):", ["Short", "Medium", "Detailed"], index=1)

if uploaded_images and len(uploaded_images) > 5:
    st.warning("⚠️ You can upload a maximum of 5 images at once.")
    uploaded_images = uploaded_images[:5]

# --- Summarize Button ---
if st.button("✨ Summarize"):
    text = ""

    # Process document files
    if uploaded_docs:
        pdfs = [f for f in uploaded_docs if f.name.endswith(".pdf")]
        docs = [f for f in uploaded_docs if f.name.endswith(".docx")]
        txts = [f for f in uploaded_docs if f.name.endswith(".txt")]

        for file in pdfs:
            text += extract_text_from_pdf(file) + "\n"
        for file in docs:
            text += extract_text_from_docx(file) + "\n"
        for file in txts:
            text += file.read().decode("utf-8") + "\n"

    # Process image files
    if uploaded_images:
        st.info(f"Extracting text from {len(uploaded_images)} image(s)...")
        text += extract_text_from_images(uploaded_images) + "\n"

    # Add manually entered text
    if manual_text.strip():
        text += "\n" + manual_text.strip()

    # Validate input
    if not text.strip():
        st.warning("⚠️ Please upload a file or enter text to summarize.")
    else:
        with st.spinner("🧠 Generating summary... Please wait..."):
            if is_resume(text):
                st.subheader("📌 Resume Summary")
                st.write(text)
            else:
                summary = summarize_large_text(text, summary_length=summary_length)
                st.subheader("📌 Document Summary")
                st.write(summary)


