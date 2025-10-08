# 🤖 AI Summarizer

An intelligent text summarization web app built with **Streamlit** and **Hugging Face Transformers**.  
It can summarize text from **PDFs, Word Documents, Text Files, or Images**, and even works with manually pasted text.

---

## 🚀 Features

- 🧠 **AI-Powered Summarization:** Uses BART Transformer for high-quality summaries
- 📄 **Multiple Input Formats:** PDF, DOCX, TXT, Image (OCR), or manual text input
- 🖼️ **OCR Support:** Extracts text from up to 5 uploaded images
- ⚙️ **Custom Summary Length:** Choose between **Short**, **Medium**, or **Detailed** output
- 🧾 **Resume Detection:** Automatically identifies and summarizes resumes separately
- 💾 **Streamlit Interface:** Simple, fast, and interactive web UI
- 🧑‍💻 **Developed by Meghana Arvapally**

---

## 🧰 Tech Stack

- **Python 3.9+**
- **Streamlit**
- **Transformers (Hugging Face)**
- **PyTorch**
- **pdfplumber**
- **python-docx**
- **pytesseract**
- **Pillow**
- **Sumy**
- **NLTK**

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ai-summarizer.git
   cd ai-summarizer
   Install dependencies
   ```

bash
Copy code
pip install -r requirements.txt
Install Tesseract OCR

Windows:
Download from Tesseract OCR Installer
(default path: C:\Program Files\Tesseract-OCR\tesseract.exe)

Linux / macOS:

bash
Copy code
sudo apt install tesseract-ocr
