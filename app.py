import streamlit as st
import joblib
import re
import html

from pypdf import PdfReader
from docx import Document


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered"
)


# -----------------------------
# Same preprocessing used
# during model training
# -----------------------------
def clean_resume(text):
    text = html.unescape(str(text))

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep useful characters such as +, #, . and -
    text = re.sub(r"[^a-zA-Z0-9+#.\- ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    data = joblib.load("resume_classifier.pkl")

    tfidf = data["tfidf"]
    model = data["model"]

    return tfidf, model


tfidf, model = load_model()


# -----------------------------
# Extract text from file
# -----------------------------
def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    # TXT
    if file_name.endswith(".txt"):
        return uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

    # PDF
    elif file_name.endswith(".pdf"):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    # DOCX
    elif file_name.endswith(".docx"):

        document = Document(uploaded_file)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    return ""


# -----------------------------
# UI
# -----------------------------
st.title("📄 Resume Classification System")

st.write(
    "Upload a resume and the model will predict its career category."
)

st.info(
    "Supported formats: PDF, DOCX and TXT"
)


uploaded_file = st.file_uploader(
    "Upload your Resume",
    type=["pdf", "docx", "txt"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔍 Predict Category"):

        with st.spinner("Analyzing resume..."):

            # Extract text
            resume_text = extract_text(uploaded_file)

            if not resume_text.strip():

                st.error(
                    "Could not extract text from this resume."
                )

            else:

                # Same preprocessing as training
                cleaned_text = clean_resume(resume_text)

                # TF-IDF transformation
                features = tfidf.transform(
                    [cleaned_text]
                )

                # SVM prediction
                prediction = model.predict(features)[0]

                st.success("Prediction completed!")

                st.subheader("Predicted Category")

                st.markdown(
                    f"## 🎯 {prediction}"
                )

                # Show extracted text
                with st.expander("View Extracted Resume Text"):

                    st.write(
                        resume_text[:5000]
                    )