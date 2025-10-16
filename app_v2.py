# app_v2.py
# AI Summarizer (updated) — safe OCR + structured resume output
import streamlit as st
from transformers import pipeline
import pdfplumber
import docx
import pytesseract
from PIL import Image
import nltk
import shutil
import platform
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import re

# -------------------------
# Setup & Downloads
# -------------------------
nltk.download('punkt', quiet=True)

# Configure Tesseract safely (works on Windows, Linux if tesseract installed)
pytesseract_available = True
if platform.system() == "Windows":
    # Keep Windows default path, but only if binary exists there
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
    elif shutil.os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd = win_path
    else:
        pytesseract_available = False
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
else:
    pytesseract_available = False

# -------------------------
# Model loading (cache)
# -------------------------
@st.cache_resource
def load_summarizer():
    # Adjust device_map/torch usage when GPU is available — default pipeline handles CPU
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# -------------------------
# Helper functions
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        # fallback try with pypdf (if pdfplumber fails)
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            for p in reader.pages:
                text += p.extract_text() or ""
        except Exception:
            pass
    return text.strip()

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception:
        return ""

def extract_text_from_images(images):
    if not pytesseract_available:
        st.warning("⚠ OCR is not available on this environment. Image summarization is disabled here.")
        return ""
    text = ""
    progress = st.progress(0)
    total = len(images)
    for idx, img in enumerate(images):
        try:
            image = Image.open(img)
            text_chunk = pytesseract.image_to_string(image, config="--psm 6")
            text += text_chunk + "\n"
        except Exception as e:
            # continue on failures (do not crash)
            st.write(f"⚠ Could not process image {getattr(img, 'name', idx)}: {e}")
        progress.progress((idx + 1) / total)
        time.sleep(0.15)
    progress.empty()
    return text.strip()

def extractive_summary(text, sentences_count=5):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer_lsa = LsaSummarizer()
        summary = summarizer_lsa(parser.document, sentences_count)
        return " ".join(str(s) for s in summary)
    except Exception:
        # fallback: return first sentences
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sents[:sentences_count])

def chunked_abstractive_summary(text, summary_length="Medium"):
    # chunking by words to avoid token limit issues
    words = text.split()
    if len(words) == 0:
        return ""
    chunk_size = 800  # ~ safe for BART
    length_settings = {
        "Short": {"max": 80, "min": 30},
        "Medium": {"max": 150, "min": 60},
        "Detailed": {"max": 250, "min": 100}
    }
    settings = length_settings.get(summary_length, length_settings["Medium"])
    max_len = settings["max"]
    min_len = settings["min"]

    summaries = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            try:
                out = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                summaries.append(out[0]['summary_text'])
            except Exception:
                # fallback to extractive for this chunk
                summaries.append(extractive_summary(chunk, sentences_count=3))

    combined = " ".join(summaries)
    # If combined is still long, compress once
    if len(combined.split()) > chunk_size:
        try:
            final = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
            return final[0]['summary_text']
        except Exception:
            return combined
    return combined

# Resume parsing: find sections by heading keywords and extract following lines
def parse_resume_sections(text):
    # Normalize headings and split into lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Combine into a single text with line breaks for regex scanning
    joined = "\n".join(lines)

    # Common resume headings (case-insensitive)
    headings = [
        "professional summary", "summary", "objective", "experience",
        "work experience", "education", "skills", "technical skills",
        "projects", "certifications", "achievements", "awards", "interests"
    ]
    # Build regex to find headings
    pattern = r'(?im)^(?:' + r'|'.join(re.escape(h) for h in headings) + r')\s*[:\-–]?\s*$'
    # Find heading positions
    matches = list(re.finditer(pattern, joined, flags=re.MULTILINE))
    sections = {}

    if matches:
        # capture region between headings
        for idx, m in enumerate(matches):
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(joined)
            heading = joined[m.start():m.end()].strip().lower().rstrip(':').strip()
            heading = re.sub(r'[:\-–\s]+$', '', heading)
            content = joined[start:end].strip()
            # clean content lines
            content = "\n".join([ln.strip() for ln in content.splitlines() if ln.strip()])
            sections[heading.title()] = content
    else:
        # try heuristic: look for "Name" line and emails/phones etc
        # fallback: attempt to extract bullets of skills via keywords
        # We'll still return empty to signal structured parse failure
        return {}

    return sections

def pretty_print_resume_sections(sections):
    out = []
    for heading, content in sections.items():
        out.append(f"**{heading}:**\n{content}\n")
    return "\n".join(out)

def is_resume(text):
    if not text or len(text.strip().split()) < 30:
        return False
    keywords = ["experience", "education", "skills", "projects", "certifications", "achievements"]
    match_count = sum(1 for k in keywords if k.lower() in text.lower())
    # require at least 3 matches and reasonably short (< 2000 words) to be considered resume
    return match_count >= 3 and len(text.split()) < 2500

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("🤖 AI Summarizer")
st.write("Summarize PDFs, Word Docs, Text Files, Images (OCR) or paste text. Resume detection outputs structured resume summary.")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    uploaded_docs = st.file_uploader(
        "📄 Upload Documents (PDF, DOCX, TXT) — multiple allowed",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
with col2:
    uploaded_images = st.file_uploader(
        "🖼️ Upload up to 5 Images (JPG, JPEG, PNG) — for OCR",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

if uploaded_images and len(uploaded_images) > 5:
    st.warning("Please upload a maximum of 5 images.")
    uploaded_images = uploaded_images[:5]

manual_text = st.text_area("📝 Or paste text here manually:", height=220)
summary_length = st.selectbox("Select summary length (applies to normal documents):", ["Short", "Medium", "Detailed"], index=1)

# Summarize button (no auto-run)
if st.button("✨ Summarize"):
    text = ""

    # Documents
    if uploaded_docs:
        for f in uploaded_docs:
            name = f.name.lower()
            if name.endswith(".pdf"):
                text += extract_text_from_pdf(f) + "\n"
            elif name.endswith(".docx"):
                text += extract_text_from_docx(f) + "\n"
            elif name.endswith(".txt"):
                try:
                    text += f.read().decode("utf-8") + "\n"
                except Exception:
                    try:
                        text += f.getvalue().decode("utf-8") + "\n"
                    except Exception:
                        pass

    # Images (OCR)
    if uploaded_images:
        st.info(f"Extracting text from {len(uploaded_images)} image(s)...")
        text += extract_text_from_images(uploaded_images) + "\n"

    # Manual text
    if manual_text and manual_text.strip():
        text += "\n" + manual_text.strip()

    if not text.strip():
        st.warning("⚠ Please upload files or paste text before summarizing.")
    else:
        with st.spinner("🧠 Generating summary..."):
            # If resume detected → structured resume summary
            if is_resume(text):
                st.subheader("📌 Resume Summary")
                sections = parse_resume_sections(text)
                if sections:
                    # print structured sections
                    for heading, content in sections.items():
                        st.markdown(f"**{heading}**")
                        # show content preserving line breaks
                        st.text(content)
                        st.markdown("")  # spacing
                else:
                    # fallback: create extractive/resume-style summary
                    st.info("Structured sections not detected clearly — using extractive-style resume summary.")
                    resume_summary = extractive_summary(text, sentences_count=10)
                    st.write(resume_summary)
            else:
                # regular document summarization (abstractive chunked)
                summary = chunked_abstractive_summary(text, summary_length=summary_length)
                st.subheader("📌 Document Summary")
                st.write(summary)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Developed by <b>Meghana Arvapally</b> © 2025 &nbsp; · &nbsp; Powered by <b>Hugging Face Transformers</b> & <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True
)
