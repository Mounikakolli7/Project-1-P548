import streamlit as st
import pickle
from docx import Document
from PyPDF2 import PdfReader

# --------------------------
# Load saved model & vectorizer
# --------------------------
MODEL_PATH = "model.pkl"
VEC_PATH = "vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VEC_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Resume Classifier", layout="wide")
st.title("📄 Entry-Level Resume Classifier")

resume_input = st.text_area("Paste resume text OR upload a file below:")

uploaded_file = st.file_uploader("Upload resume file", type=["docx", "pdf", "txt"])

# --- Handle Uploaded File ---
if uploaded_file is not None:
    if uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        resume_input = "\n".join(p.text for p in doc.paragraphs)
    elif uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        resume_input = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif uploaded_file.name.endswith(".txt"):
        resume_input = uploaded_file.read().decode("utf-8")

# --- Prediction ---
if st.button("Predict"):
    if resume_input.strip() == "":
        st.warning("⚠️ Please enter or upload resume text.")
    else:
        vec = vectorizer.transform([resume_input])
        pred = model.predict(vec)[0]
        st.success(f"✅ Predicted Category: **{pred}**")
