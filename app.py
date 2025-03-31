import streamlit as st
import tempfile
import os
from document_loader import extract_text
from vector_store import create_vector_store
from qa_engine import answer_question
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="📄 AI Document Assistant")

st.title("📄 AI Document Assistant")
st.markdown("Upload a document (PDF, Word, or Text), ask questions based on its content, and get instant answers using OpenAI GPT.")

# Step 1: Upload file
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    st.success("File uploaded successfully.")

    # Step 2: Extract text
    with st.spinner("Extracting text from the document..."):
        text = extract_text(temp_file_path)
        st.text_area("Extracted Text", text, height=200)

    # Step 3: Create vector store
    with st.spinner("Indexing document..."):
        vector_store = create_vector_store(text)

    # Step 4: Ask a question
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Generating answer..."):
            response = answer_question(vector_store, question)
            st.markdown(f"### 💬 Answer:\n{response}")

    # Cleanup temp file
    os.remove(temp_file_path)
