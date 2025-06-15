import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# load SBERT once
SBERT = SentenceTransformer('all-MiniLM-L6-v2')

def _ensure_str(x):
    return " ".join(x) if isinstance(x, list) else str(x)

def dynamic_keyword_overlap(resume: str, jd: str, top_k: int = 50) -> float:
    # build TF-IDF on *both* documents and allow rare terms
    vec = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1              # <-- allow terms that appear just once
    )
    # fit on both JD and resume
    X = vec.fit_transform([jd, resume]).toarray()
    jd_vec = X[0]            # JD’s tfidf vector
    terms = vec.get_feature_names_out()

    # pick top_k highest-scoring JD-terms
    top_idx   = np.argsort(jd_vec)[-top_k:]
    top_terms = { terms[i] for i in top_idx }

    # count how many of those appear in the resume
    resume_tokens = set(resume.lower().split())
    hits = sum(1 for t in top_terms if t in resume_tokens)

    return hits / top_k if top_k else 0.0


def bm25_score(resume: str, jd: str) -> float:
    resume_tokens = resume.lower().split()
    jd_tokens = jd.lower().split()
    bm25 = BM25Okapi([resume_tokens])
    scores = bm25.get_scores(jd_tokens)
    # normalize by max possible
    max_score = max(scores) if len(scores) > 0 else 1.0
    return float(np.mean(scores) / max_score)

def score_resume_against_job(
    resume_text,
    job_text,
    weights: Tuple[float,float,float,float] = (0.15, 0.5, 0.2, 0.15),
    thresholds: Tuple[float,float] = (0.65, 0.45),
    top_k: int = 50
) -> Tuple[float,float,float,str]:
    # ensure strings
    resume = _ensure_str(resume_text)
    jd     = _ensure_str(job_text)

    # 1) TF-IDF similarity
    vect = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,3),
        max_features=10000,
        min_df=2
    )
    vecs = vect.fit_transform([resume, jd])
    tfidf_sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])

    # 2) SBERT embedding similarity
    embs = SBERT.encode([resume, jd], convert_to_tensor=True)
    bert_sim = float(util.pytorch_cos_sim(embs[0], embs[1])[0][0].item())

    # 3) BM25 similarity
    bm25_sim = bm25_score(resume, jd)

    # 4) Keyword overlap
    kw_sim = dynamic_keyword_overlap(resume, jd, top_k=top_k)

    # 5) Final weighted score
    w_tfidf, w_bert, w_bm25, w_kw = weights
    final_score = (
        w_tfidf * tfidf_sim +
        w_bert  * bert_sim +
        w_bm25  * bm25_sim +
        w_kw    * kw_sim
    )

    # 6) Label
    strong, moderate = thresholds
    if final_score >= strong:
        label = "Strong"
    elif final_score >= moderate:
        label = "Moderate"
    else:
        label = "Weak"

    return (
        round(tfidf_sim, 4),
        round(bert_sim, 4),
        round(final_score, 4),
        label
    )

