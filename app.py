import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from groq import Groq
from transformers import AutoTokenizer, AutoModel
import torch
import textwrap

# ------------------- CONFIG -------------------
COLLECTION_NAME = "sc_judgments_2025_legalbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ------------------- LOAD ENV -------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_")
QDRANT_URL = os.getenv("QDRANT_URL", "...")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "...")

if not GROQ_API_KEY:
    st.error("❌ Please set GROQ_API_KEY environment variable before running.")
    st.stop()

# ------------------- INIT CLIENTS -------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq = Groq(api_key=GROQ_API_KEY)

# Load embedder (LegalBERT)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    model = AutoModel.from_pretrained("law-ai/InLegalBERT").to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
    return emb[0].cpu().numpy().tolist()

# ------------------- QDRANT RETRIEVAL -------------------
def retrieve(query_vector, top_k=10, filters=None):
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=filters,
        with_payload=True
    ).points

# ------------------- PROMPT TEMPLATE -------------------
PROMPT_TEMPLATE = textwrap.dedent("""
You are an expert Indian legal assistant.
Using the CONTEXT below, draft a 3-paragraph summary that answers the USER QUESTION.
Include relevant case names and reasoning in plain English. Use [n] for citations.

CONTEXT:
{context_block}

USER QUESTION:
{user_question}

INSTRUCTIONS:
1. Summarize in 3 paragraphs, clear English.
2. Use citations [n] referencing cases in the CONTEXT.
3. End with "Sources:" listing case names.
""").strip()

# ------------------- RAG + LLM -------------------
def build_context(points, char_limit=6000):
    parts, sources = [], []
    total_chars = 0
    for i, p in enumerate(points):
        payload = p.payload or {}
        preview = payload.get("text_preview") or payload.get("chunk_text") or ""
        case = payload.get("case_name", payload.get("file_name", "Unknown Case"))
        yr = payload.get("year", "")
        entry = f"[{i+1}] {case} ({yr})\n{preview}\n"
        if total_chars + len(entry) > char_limit:
            break
        parts.append(entry)
        sources.append((i+1, case, yr))
        total_chars += len(entry)
    return "\n".join(parts), sources

def generate_summary(user_question, query_vector, filters=None):
    points = retrieve(query_vector, filters=filters)
    if not points:
        return "No matching results found in Qdrant.", []
    context, sources = build_context(points)
    prompt = PROMPT_TEMPLATE.format(context_block=context, user_question=user_question)
    completion = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": "You are a helpful legal research assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=800
    )
    return completion.choices[0].message.content, sources

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Legal Assistant (Qdrant + Groq)", layout="wide")
st.title("⚖️ Legal Research Assistant")
st.markdown("Use Indian Supreme Court judgments (Qdrant DB) + Groq LLM summarization")

with st.sidebar:
    st.header("🔍 Search Filters")
    jurisdiction = st.text_input("Jurisdiction", "India")
    case_name = st.text_input("Case name contains (optional)")
    year_min = st.number_input("From year", min_value=1950, max_value=2025, value=2020)
    year_max = st.number_input("To year", min_value=1950, max_value=2025, value=2025)
    top_k = st.slider("Top K results", 3, 20, 10)

user_query = st.text_area("Enter your legal question:", placeholder="e.g., Summarize Supreme Court judgments on data privacy and proportionality")

if st.button("Generate Summary"):
    with st.spinner("Retrieving and reasoning with Groq..."):
        query_vec = embed_text(user_query)
        must_filters = [FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction))]
        if case_name:
            must_filters.append(FieldCondition(key="case_name", match=MatchValue(value=case_name)))
        must_filters.append(FieldCondition(key="year", range=Range(gte=year_min, lte=year_max)))
        filters = Filter(must=must_filters)
        summary, sources = generate_summary(user_query, query_vec, filters)
        st.subheader("🧾 Generated Summary")
        st.write(summary)

        if sources:
            st.markdown("### 📚 Sources")
            for idx, case, yr in sources:
                st.markdown(f"**[{idx}]** {case} ({yr})")

st.markdown("---")
st.caption("Built with Qdrant + Groq | LegalBERT embeddings | Demo app © 2025")
