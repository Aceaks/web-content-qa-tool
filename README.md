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
### Clone the repository
git clone https://github.com/Aceaks/web-content-qa-tool
cd web-content-qa-tool

## Install Dependencies & Ollama
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

## Run the Application
### In a separate terminal window, start the Streamlit app
streamlit run app.py

** The .devcontainer folder is not required and got created while trying to generate live hosting of the application!
