import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

st.set_page_config(page_title="PolicyBot", layout="wide")
st.title("🤖 PolicyBot - Internal Policy Assistant")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_texts = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                loader = PyPDFLoader(tmp_file.name)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = text_splitter.split_documents(documents)
                all_texts.extend(texts)

        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(all_texts, embeddings, persist_directory="index_store")
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever, return_source_documents=True)

        query = st.text_input("Ask a policy-related question")

        if query:
            with st.spinner("Generating answer..."):
                result = qa(query)
                answer = result['result']
                sources = result['source_documents']

                st.session_state.history.append((query, answer, sources))

# Display chat history
if st.session_state.get("history"):
    st.subheader("📝 Chat History")
    for q, a, s in st.session_state.history:
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 PolicyBot:** {a}")
        with st.expander("📄 View Sources"):
            for doc in s:
                st.markdown(f"- **Page Content:** {doc.page_content[:300]}...")

        st.code(a, language='text')

# Footer
st.markdown("---")
st.caption("Powered by LangChain, OpenAI, and ChromaDB")
