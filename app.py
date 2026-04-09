import streamlit as st

# 2. Import your analyze_resume function from skills.py
from skills import analyze_resume

import pdfplumber


# 3. Show a title: "Smart Resume Analyzer"
st.title("Smart Resume Analyzer")

# 4. Show a subtitle describing what the app does (one line)
st.write(
    "Paste your resume below and click 'Analyze Resume' to get insights.")


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text


tab1, tab2 = st.tabs(["📄 Upload PDF", "📝 Paste Text"])

# Initialize both variables as empty
pdf_text = ""
pasted_text = ""

with tab1:
    # PDF code here — this appears only on tab 1
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF format)", type=["pdf"])
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF uploaded and text extracted successfully!")
with tab2:
    # Text area code here — this appears only on tab 2
    pasted_text = st.text_area("Or paste your resume text here", height=300)


# 5. Create a text_area where user can paste their resume
resume_text = pdf_text if pdf_text else pasted_text

# straight to the button — no warning block between them
if st.button("Analyze Resume"):
    if resume_text is None or resume_text.strip() == "":
        st.warning("Please paste your resume or upload a PDF before analyzing.")
    else:
        result = analyze_resume(resume_text)
        # c. Show "Predicted Category:" + the category
        st.subheader("🎯 Predicted Category")
        st.success(result['predicted_category'])

        st.subheader("✅ Skills Found")
        st.info(', '.join(result['skills_found']))

        st.subheader("⚠️ Skills Missing")
        st.warning(', '.join(result['skills_missing']))
