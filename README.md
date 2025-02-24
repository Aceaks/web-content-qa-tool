# 🌐 Web Content Q&A Tool

## 📌 Overview
This **Web Content Q&A Tool** allows users to:
- Enter URLs to extract web content.
- Ask questions about the extracted content.
- Get answers using **Ollama LLM & ChromaDB** for complete local inferences.

## 🚀 Features
✅ **Extracts text from URLs**  
✅ **Retrieves information using ChromaDB**  
✅ **Uses Ollama for local inference**  
✅ **Simple & fast UI with Streamlit**

---

## 🛠 Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/Aceaks/web-content-qa-tool
cd web-content-qa-tool
### 2️⃣  Installation
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
### 3️⃣ Running app
In seperate terminal window run the app by following command:
streamlit run app.py
