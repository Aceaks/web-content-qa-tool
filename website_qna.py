
# import streamlit as st
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaLLM
# from langchain.prompts import PromptTemplate

# # Streamlit UI
# st.title("🌐 Web Content Q&A Tool")
# st.markdown("## 🤖 Enter URLs & Ask Questions")

# # User input for URLs
# url_input = st.text_area("📜 **Enter URLs (one per line):**", "")
# urls = url_input.strip().split("\n") if url_input else []

# # Button to ingest content
# if st.button("🔄 **Ingest URLs**"):
#     if urls:
#         st.session_state["urls"] = urls
#         st.session_state["ingested"] = False
#         st.success("✅ URLs stored! Now click **'Process Content'** to extract information.")
#     else:
#         st.error("❌ Please enter at least one URL.")

# # Process URLs
# if "urls" in st.session_state and not st.session_state.get("ingested", False):
#     with st.spinner("⏳ **Processing URLs... Please wait**"):
#         loader = UnstructuredURLLoader(urls=st.session_state["urls"])
#         documents = loader.load()

#         # Split text into manageable chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
#                                                        chunk_overlap=150)
#         text_chunks = []
#         metadatas = []

#         for doc in documents:
#             chunks = text_splitter.split_text(doc.page_content)
#             text_chunks.extend(chunks)
#             metadatas.extend([{"source": doc.metadata["source"]}] * len(chunks))

#         # Initialize embeddings and ChromaDB
#         embeddings = HuggingFaceEmbeddings()
#         index = Chroma.from_texts(text_chunks, embeddings, metadatas=metadatas,
#                                   persist_directory="./chroma_db")

#         st.session_state["index"] = index
#         st.session_state["ingested"] = True
#         st.success("✅ **Content processed successfully!** You can now ask questions.")

# # Question input
# query = st.text_input("🔍 **Enter your question:**")

# # Perform Q&A
# if st.button("💡 **Get Answer**") and query:
#     if "index" in st.session_state:
#         # Initialize Ollama LLM
#         llm = OllamaLLM(model="llama3.2")

#         # Define a strong retrieval-based prompt
#         template = """You are an AI answering based **only** on the given context. 
#         If you cannot find the answer, say "I don't know."

#         Context: {context}

#         Question: {question}

#         Answer:
#         """
#         qa_chain_prompt = PromptTemplate.from_template(template)

#         # Set up the RetrievalQA chain
#         retriever = st.session_state["index"].as_retriever(search_type="similarity", search_kwargs={"k": 10})
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type_kwargs={"prompt": qa_chain_prompt}
#         )

#         # Get the answer
#         result = qa_chain.invoke({"query": query})
#         answer = result["result"]

#         # Display answer
#         st.markdown("## 💡 **Answer**")
#         st.success(answer)
#     else:
#         st.error("❌ Please ingest content first by entering URLs.")

import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Streamlit UI
st.title("🌐 Web Content Q&A Tool")
st.markdown("## 🤖 Enter URLs & Ask Questions")

# User input for URLs
url_input = st.text_area("📜 **Enter URLs (one per line):**", "")
urls = url_input.strip().split("\n") if url_input else []

# Button to ingest content
if st.button("🔄 **Ingest URLs**"):
    if urls:
        st.session_state["urls"] = urls
        st.session_state["ingested"] = False
        st.success("✅ URLs stored! Now click **'Process Content'** to extract information.")
    else:
        st.error("❌ Please enter at least one URL.")

# Process URLs
if "urls" in st.session_state and not st.session_state.get("ingested", False):
    with st.spinner("⏳ **Processing URLs... Please wait**"):
        loader = UnstructuredURLLoader(urls=st.session_state["urls"])
        documents = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        text_chunks = []
        metadatas = []

        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            text_chunks.extend(chunks)
            metadatas.extend([{"source": doc.metadata["source"]}] * len(chunks))

        # Initialize embeddings and FAISS (instead of ChromaDB)
        embeddings = HuggingFaceEmbeddings()
        index = FAISS.from_texts(text_chunks, embeddings)

        st.session_state["index"] = index
        st.session_state["ingested"] = True
        st.success("✅ **Content processed successfully!** You can now ask questions.")

# Question input
query = st.text_input("🔍 **Enter your question:**")

# Perform Q&A
if st.button("💡 **Get Answer**") and query:
    if "index" in st.session_state:
        # Initialize Ollama LLM
        llm = OllamaLLM(model="llama3.2")

        # Define a strong retrieval-based prompt
        template = """You are an AI answering based **only** on the given context. 
        If you cannot find the answer, say "I don't know."

        Context: {context}

        Question: {question}

        Answer:
        """
        qa_chain_prompt = PromptTemplate.from_template(template)

        # Set up the RetrievalQA chain
        retriever = st.session_state["index"].as_retriever(search_kwargs={"k": 10})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_chain_prompt}
        )

        # Get the answer
        result = qa_chain.invoke({"query": query})
        answer = result["result"]

        # Display answer
        st.markdown("## 💡 **Answer**")
        st.success(answer)
    else:
        st.error("❌ Please ingest content first by entering URLs.")
