import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from model import score_resume_against_job

st.set_page_config(page_title="Resume Matcher", layout="centered")

# Inject CSS for styling sidebar buttons and centering navigation title
st.markdown(
    """
    <style>
    /* Equal width buttons with pastel hover */
    [data-testid="stSidebar"] button {
        width: 100% !important;
        margin-bottom: 5px;
        transition: background-color 0.2s ease;
        border: none;
        background-color: transparent;
        padding: 8px 0;
        text-align: left;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #b2ebf2 !important; /* darker pastel blue */
    }
    /* Center the navigation title */
    [data-testid="stSidebar"] h3 {
        text-align: center !important;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation Bar")  # Title centered via injected CSS
st.sidebar.markdown("---")

# initialize session state for page selection
def init_page():
    if 'page' not in st.session_state:
        st.session_state['page'] = "🏠 Introduction"
init_page()

# navigation buttons
if st.sidebar.button("🏠 Introduction"):
    st.session_state['page'] = "🏠 Introduction"
if st.sidebar.button("📘 How to Use"):
    st.session_state['page'] = "📘 How to Use"
if st.sidebar.button("🤖 Resume Matcher"):
    st.session_state['page'] = "🤖 Resume Matcher"
if st.sidebar.button("📖 About"):
    st.session_state['page'] = "📖 About"

# set current page
page = st.session_state['page']

# --- Introduction Page ---
if page == "🏠 Introduction":
    st.title("📄 Resume–Job Match Scorer")
    st.subheader("🔍 A Machine Learning–Powered Resume Evaluator")
    st.markdown("""
    Welcome to **Resume–Job Match Scorer**, your go-to tool for instantly assessing how well a candidate’s resume aligns with a job description. Leveraging state-of-the-art NLP and embedding techniques, our app provides:

    1. **TF-IDF Similarity** – Measures overlap of key terms and phrases essential to the role.
    2. **SBERT Embedding Similarity** – Captures deeper semantic meaning using a lightweight Sentence-BERT model for high accuracy.
    3. **BM25 Relevance Scoring** – Ranks document relevance by balancing term frequency and document length.
    4. **Dynamic Keyword Overlap** – Extracts top job-specific keywords and highlights matches in the resume.

    ### 🚀 How It Works
    - **Step 1: Upload**  
      Drag and drop or browse to upload your resume(s) (PDF or plain text) and paste your job description.
    - **Step 2: Analyze**  
      Click **Match Resumes**. Our pipeline will:
        - Parse and clean text.
        - Compute each similarity metric.
        - Aggregate into a final weighted score.
    - **Step 3: Review Results**  
      See per-resume scores and labels (Strong/Moderate/Weak), plus an interactive breakdown of each metric.

    ### 🔧 Under the Hood
    - **TF-IDF**: Uses n-grams (1–3) with min_df filtering to focus on meaningful terms.
    - **SBERT**: Utilizes the `all-MiniLM-L6-v2` model for efficient semantic embeddings.
    - **BM25**: Powered by `rank_bm25` for robust relevance ranking.
    - **Keyword Overlap**: Selects top-K terms via TF-IDF on the JD and checks presence in the resume.

    ### ⚙️ Created By:
    - **A.B. Ghalib**  
    - **Zohha Azhar**  
    - **Laiba Arshad**
    """)
    st.markdown("---")
    st.info("Tip: Ensure your text is machine-readable—avoid scanned images or complex layouts.")

# --- Instructions Page ---
elif page == "📘 How to Use":
    st.title("🛠️ How to Use This Web App")
    st.markdown("""
    ### 📌 Step-by-Step Instructions

    1. **Select the 'Upload & Score' tab** from the sidebar.
    2. **Paste your Job Description** into the provided text area. Use clear, bulleted format.
    3. **Upload one or more resumes** in PDF or .txt format. Drag-and-drop is supported.
    4. **Click 'Match Resumes'** to start the evaluation.
    5. **Review the output table**, which displays:
       - **TF-IDF Score**: Keyword and phrase overlap.
       - **BERT Score**: Semantic similarity rating.
       - **BM25 Score**: Relevance ranking score.
       - **Keyword Overlap**: Percentage of top-K JD terms found in the resume.
       - **Final Score**: Weighted combination of all metrics.
       - **Match Label**: Categorizes into **Strong**, **Moderate**, or **Weak**.

    ### 📁 Example Output Table:
    | File Name          | TF-IDF | BERT  | BM25  | Overlap | Final | Match    |
    |--------------------|--------|-------|-------|---------|-------|----------|
    | resume_sara.pdf    | 0.85   | 0.92  | 0.78  | 0.90    | 0.88  | Strong   |
    | resume_ahmed.pdf   | 0.45   | 0.50  | 0.40  | 0.55    | 0.48  | Moderate |
    | resume_fatima.pdf  | 0.20   | 0.15  | 0.10  | 0.25    | 0.17  | Weak     |
    """)

# --- Model Page ---
elif page == "🤖 Resume Matcher":
    st.title("🤖 Resume Matching Engine")
    job_description = st.text_area("📄 Paste Job Description", height=180)

    uploaded_files = st.file_uploader(
        "📁 Upload Resume PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )

    if st.button("🚀 Match Resumes") and job_description.strip() and uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            try:
                reader = PdfReader(uploaded_file)
                texts = [p.extract_text() for p in reader.pages if p.extract_text()]
                if not texts:
                    raise ValueError("No extractable text found in this PDF.")
                text = " ".join(texts)

                tfidf_score, bert_score, final_score, label = score_resume_against_job(
                    text, job_description
                )

                results.append({
                    "File Name": uploaded_file.name,
                    "TF-IDF Score": round(tfidf_score, 2),
                    "BERT Score": round(bert_score, 2),
                    "Final Score": round(final_score, 2),
                    "Match": label
                })
            except Exception as e:
                results.append({
                    "File Name": uploaded_file.name,
                    "TF-IDF Score": "Error",
                    "BERT Score": "Error",
                    "Final Score": "Error",
                    "Match": str(e)
                })

        df = pd.DataFrame(results)
        st.markdown("### 📊 Match Results")
        st.dataframe(df)

# --- About Page ---
elif page == "📖 About":
    st.title("📖 About & FAQs")
    st.markdown("""
    **Q1: What file formats are supported?**  
    A: We support PDF and plain-text (.txt) resume uploads.

    **Q2: How is the final score calculated?**  
    A: It's a weighted combination of TF-IDF, SBERT embedding similarity, BM25 relevance, and dynamic keyword overlap.

    **Q3: What do the match labels mean?**  
    - **Strong**: Candidate score ≥ 0.65  
    - **Moderate**: 0.45 ≤ score < 0.65  
    - **Weak**: score < 0.45

    **Q4: Why might my resume be scored 'Weak'?**  
    A: If key skills or job-specific keywords are missing or your resume text isn't well-aligned to the JD.

    **Q5: How can I improve accuracy?**  
    - Use clear, bullet-pointed job descriptions.  
    - Ensure resumes are text-based (no images).  
    - Include relevant skills and experience in your resume.

    **Q6: Who can I contact for support?**  
    A: Email **A.B. Ghalib** at bsdsf22m047@pucit.edu.pk or **Zohha Azhar** at bsdsf22m042@example.com.
    """)
    st.info("Need further assistance? Reach out via the contact emails above.")
