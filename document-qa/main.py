import streamlit as st
from pdfminer.high_level import extract_text
import torch
from transformers import pipeline

# Load QA model
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = load_model()

def extract_text_from_pdf(up_file):
    text = extract_text(up_file)
    return text

st.title("_Document_  :red[Question] :green[Answer]")
st.markdown("Upload Document &mdash; Ask Question to your document")
uploaded_file = st.file_uploader("Upload a PDF File", type="pdf")

if uploaded_file and "pdf_text" not in st.session_state:
    with st.spinner("Extracting text from PDF..."):
        st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted successfully!")

if "pdf_text" in st.session_state:
    question = st.text_input("Ask a question related to the PDF content:")
    if question:
        with st.spinner("Finding the answer..."):
            result = qa_model(question=question, context=st.session_state.pdf_text)
            st.success(f"**Answer:** {result['answer']} (score: {result['score']:.2f})")            
