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

🛠 Installation & Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/Aceaks/web-content-qa-tool
cd web-content-qa-tool
2️⃣ Install Dependencies & Ollama
bash
Copy
Edit
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
3️⃣ Run the Application
bash
Copy
Edit
# In a separate terminal window, start the Streamlit app
streamlit run app.py
