import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import requests
import json
import os
import time

# ─────────────────────────────────────────────
#  LOAD API KEYS — st.secrets (deployment) with
#  fallback to .env file (local development)
# ─────────────────────────────────────────────
def get_secret(key: str) -> str:
    """Try st.secrets first (Streamlit Cloud), then env vars (local .env)."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        pass
    # local fallback — parse .env manually
    if not hasattr(get_secret, "_loaded"):
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        get_secret._loaded = True
    return os.getenv(key, "")

SERPER_API_KEY = get_secret("SERPER_API_KEY")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")
GOOGLE_FC_KEY  = get_secret("GOOGLE_FC_API_KEY")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
HF_REPO_ID      = "PalakJaiswal2401/multilingual-fake-news-detection"
HF_FILENAME     = "improved_model_v2.pt"
BASE_MODEL_NAME = "xlm-roberta-base"   # pulled directly from HuggingFace
MAX_LENGTH      = 256
STRIDE          = 128
MAX_CHUNKS      = 5
DEVICE          = torch.device("cpu")

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
SERPER_URL = "https://google.serper.dev/search"
GFC_URL    = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Instrument Sans', sans-serif;
        background-color: #080c14 !important;
        color: #c8d0e0 !important;
    }
    .stApp { background-color: #080c14 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1400px !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1525 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid #1a2744 !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #4a6080 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1a2744 !important;
        color: #00c8b4 !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0d1525 0%, #0a1020 50%, #0d1a2e 100%);
        border: 1px solid #1a2744;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(0,200,180,0.07) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem; font-weight: 800;
        letter-spacing: -0.03em; color: #f0f4ff;
        margin: 0 0 0.4rem 0; line-height: 1.1;
    }
    .hero-title span { color: #00c8b4; }
    .hero-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; color: #4a6080;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .hero-badges { display: flex; gap: 0.6rem; margin-top: 1.2rem; flex-wrap: wrap; }
    .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem; padding: 0.25rem 0.7rem;
        border-radius: 20px; letter-spacing: 0.08em;
        text-transform: uppercase; font-weight: 500;
    }
    .badge-teal  { background: rgba(0,200,180,0.1);  color: #00c8b4; border: 1px solid rgba(0,200,180,0.2);  }
    .badge-blue  { background: rgba(99,102,241,0.1); color: #818cf8; border: 1px solid rgba(99,102,241,0.2); }
    .badge-amber { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
    .badge-red   { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.2);  }

    /* Cards */
    .card {
        background: #0d1525; border: 1px solid #1a2744;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }

    /* Result comparison boxes */
    .result-box {
        border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    }
    .result-box-real {
        background: linear-gradient(135deg, rgba(0,200,180,0.08), rgba(0,168,150,0.04));
        border: 1px solid rgba(0,200,180,0.25);
        border-left: 4px solid #00c8b4;
    }
    .result-box-fake {
        background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(220,38,38,0.04));
        border: 1px solid rgba(239,68,68,0.25);
        border-left: 4px solid #ef4444;
    }
    .result-label {
        font-family: 'Syne', sans-serif; font-size: 1.7rem;
        font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.3rem 0;
    }
    .result-real { color: #00c8b4; }
    .result-fake { color: #ef4444; }
    .result-conf {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: #4a6080; letter-spacing: 0.1em;
        text-transform: uppercase; margin-bottom: 0.6rem;
    }
    .result-reason { font-size: 0.87rem; color: #8090b0; line-height: 1.6; font-style: italic; }

    /* Result header tags */
    .result-tag {
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
        padding: 0.2rem 0.6rem; border-radius: 4px;
        text-transform: uppercase; letter-spacing: 0.1em;
        display: inline-block; margin-bottom: 0.8rem;
    }
    .tag-style { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
    .tag-llm   { background: rgba(99,102,241,0.1); color: #818cf8; border: 1px solid rgba(99,102,241,0.2); }
    .tag-final { background: rgba(0,200,180,0.1);  color: #00c8b4; border: 1px solid rgba(0,200,180,0.2); }

    /* Claims */
    .claims-wrapper { margin: 0.8rem 0 1rem 0; }
    .claim-item {
        display: flex; align-items: flex-start; gap: 0.7rem;
        padding: 0.7rem 1rem; margin-bottom: 0.5rem;
        background: rgba(99,102,241,0.05);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 8px;
    }
    .claim-num {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        color: #818cf8; min-width: 18px; padding-top: 1px;
    }
    .claim-text { font-size: 0.87rem; color: #c8d0e0; line-height: 1.5; }

    /* Input */
    .stTextArea textarea {
        background: #0d1525 !important; border: 1px solid #1a2744 !important;
        border-radius: 10px !important; color: #c8d0e0 !important;
        font-family: 'Instrument Sans', sans-serif !important;
        font-size: 0.95rem !important; padding: 1rem !important;
        transition: border-color 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: #00c8b4 !important;
        box-shadow: 0 0 0 2px rgba(0,200,180,0.1) !important;
    }
    .stTextArea label { color: #4a6080 !important; font-size: 0.75rem !important; }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00c8b4 0%, #00a896 100%) !important;
        color: #080c14 !important; border: none !important;
        border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important; font-size: 0.95rem !important;
        letter-spacing: 0.04em !important; height: 3rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px rgba(0,200,180,0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(0,200,180,0.35) !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #0a1020; border: 1px solid #1a2744;
        border-radius: 10px; padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricLabel"]  { color: #4a6080 !important; font-size: 0.7rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
    [data-testid="stMetricValue"]  { color: #f0f4ff !important; font-family: 'Syne', sans-serif !important; font-size: 1.5rem !important; font-weight: 700 !important; }

    /* Progress */
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #00c8b4, #00a896) !important; border-radius: 4px !important; }
    .stProgress > div > div            { background: #1a2744 !important; border-radius: 4px !important; }

    /* Expanders */
    .streamlit-expanderHeader {
        background: #0d1525 !important; border: 1px solid #1a2744 !important;
        border-radius: 8px !important; color: #8090b0 !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important;
    }
    .streamlit-expanderContent {
        background: #0a1020 !important; border: 1px solid #1a2744 !important;
        border-top: none !important; border-radius: 0 0 8px 8px !important;
    }

    /* Alerts */
    .stAlert { border-radius: 10px !important; border: none !important; font-size: 0.85rem !important; }
    hr { border-color: #1a2744 !important; margin: 1.5rem 0 !important; }

    /* Section headers */
    .section-header {
        font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.2em; text-transform: uppercase; color: #4a6080;
        margin-bottom: 1rem; display: flex; align-items: center; gap: 0.6rem;
    }
    .section-header::after { content: ''; flex: 1; height: 1px; background: #1a2744; }

    /* Evidence cards */
    .ev-card {
        background: #0a1020; border: 1px solid #1a2744;
        border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
    }
    .ev-title   { font-weight: 600; color: #c8d0e0; font-size: 0.88rem; margin-bottom: 0.3rem; }
    .ev-snippet { color: #6080a0; font-size: 0.82rem; line-height: 1.6; margin-bottom: 0.5rem; }
    .ev-source  { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #00c8b4; }

    /* Key status bar */
    .key-status {
        display: flex; gap: 1.5rem; padding: 0.8rem 1.2rem;
        background: #0a1020; border: 1px solid #1a2744;
        border-radius: 8px; margin-bottom: 1.5rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    }
    .key-ok   { color: #00c8b4; }
    .key-fail { color: #ef4444; }
    .key-opt  { color: #4a6080; }

    /* Pipeline layer rows */
    .layer-row {
        display: flex; align-items: center; gap: 0.8rem;
        padding: 0.6rem 0; border-bottom: 1px solid #1a2744;
        font-size: 0.85rem; color: #8090b0;
    }
    .layer-row:last-child { border-bottom: none; }
    .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .dot-teal   { background: #00c8b4; box-shadow: 0 0 6px rgba(0,200,180,0.5); }
    .dot-indigo { background: #818cf8; box-shadow: 0 0 6px rgba(129,140,248,0.5); }
    .dot-amber  { background: #fbbf24; box-shadow: 0 0 6px rgba(251,191,36,0.5); }
    .layer-name { font-family: 'Syne', sans-serif; font-weight: 600; color: #c8d0e0; min-width: 140px; font-size: 0.82rem; }

    /* Coming soon */
    .coming-soon {
        background: linear-gradient(135deg, #0d1525, #0a1020);
        border: 1px dashed #1a2744; border-radius: 16px;
        padding: 4rem 2rem; text-align: center;
    }
    .coming-soon-icon { font-size: 4rem; margin-bottom: 1rem; }
    .coming-soon-title {
        font-family: 'Syne', sans-serif; font-size: 1.8rem;
        font-weight: 800; color: #f0f4ff; margin-bottom: 0.5rem;
    }
    .coming-soon-sub { color: #4a6080; font-size: 0.9rem; line-height: 1.6; max-width: 400px; margin: 0 auto; }
    .coming-soon-badge {
        display: inline-block; margin-top: 1.5rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        padding: 0.4rem 1rem; border-radius: 20px;
        background: rgba(251,191,36,0.1); color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.2); letter-spacing: 0.1em;
    }
    .planned-features { margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
    .pf-item {
        background: #0d1525; border: 1px solid #1a2744; border-radius: 8px;
        padding: 0.8rem 1.2rem; font-size: 0.82rem; color: #8090b0; text-align: left;
        min-width: 180px;
    }
    .pf-item strong { display: block; color: #c8d0e0; font-family: 'Syne', sans-serif; margin-bottom: 0.2rem; }

    /* Comparison divider */
    .vs-divider {
        display: flex; align-items: center; justify-content: center;
        margin: 0.5rem 0 1.5rem 0;
    }
    .vs-pill {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        padding: 0.2rem 0.8rem; border-radius: 20px;
        background: #1a2744; color: #4a6080; letter-spacing: 0.1em;
    }

    h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #f0f4ff !important; letter-spacing: -0.02em !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class HierarchicalXLMRBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_MODEL_NAME)
        hidden_size  = 768
        self.doc_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8,
                dim_feedforward=2048, dropout=0.1, batch_first=True
            ), num_layers=2
        )
        self.attention  = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask):
        batch_size, num_chunks, seq_len = input_ids.size()
        input_ids      = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        outputs        = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb        = outputs.last_hidden_state[:, 0, :].view(batch_size, num_chunks, -1)
        doc_out        = self.doc_transformer(cls_emb)
        attn_w         = torch.softmax(self.attention(doc_out), dim=1)
        doc_rep        = torch.sum(attn_w * doc_out, dim=1)
        return self.classifier(doc_rep)


@st.cache_resource(show_spinner="Loading TruthLens model from HuggingFace...")
def load_model():
    # Download fine-tuned weights from HuggingFace Hub
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model      = HierarchicalXLMRBase()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
#  LAYER 1 — STYLE DETECTION
# ─────────────────────────────────────────────
def predict_article(text, model, tokenizer):
    encoding = tokenizer(
        text, max_length=MAX_LENGTH, stride=STRIDE,
        truncation=True, padding="max_length",
        return_overflowing_tokens=True, return_tensors="pt"
    )
    ids  = encoding["input_ids"]
    mask = encoding["attention_mask"]
    if ids.size(0) > MAX_CHUNKS:
        ids, mask = ids[:MAX_CHUNKS], mask[:MAX_CHUNKS]
    pad = MAX_CHUNKS - ids.size(0)
    if pad > 0:
        ids  = torch.cat([ids,  torch.zeros((pad, MAX_LENGTH), dtype=torch.long)], dim=0)
        mask = torch.cat([mask, torch.zeros((pad, MAX_LENGTH), dtype=torch.long)], dim=0)
    with torch.no_grad():
        probs = torch.softmax(model(ids.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)), dim=1)
    prob_real, prob_fake = probs[0][0].item(), probs[0][1].item()
    return ("Real" if prob_real > prob_fake else "Fake"), max(prob_real, prob_fake), prob_real, prob_fake


# ─────────────────────────────────────────────
#  GROQ LLM HELPER
# ─────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 512},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        st.error(f"Groq error {resp.status_code}: {resp.text[:200]}")
        return ""
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return ""


# ─────────────────────────────────────────────
#  CLAIM EXTRACTION
# ─────────────────────────────────────────────
def extract_claims(text: str) -> list:
    prompt = f"""You are a fact-checking assistant. Extract 3 to 5 short, specific, verifiable factual claims from the news text below. These must be concrete facts (names, numbers, events, dates, statistics) that can be verified via a web search.

Return ONLY a valid JSON array of strings. No explanation, no markdown, no preamble.
Example: ["India won the T20 World Cup 2024", "Rohit Sharma scored 76 runs", "The match was held in Barbados"]

News text:
{text[:1500]}"""
    raw = call_llm(prompt)
    if not raw:
        return [text[:200]]
    try:
        clean  = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        claims = json.loads(clean)
        if isinstance(claims, list) and claims:
            return [str(c) for c in claims[:5]]
    except Exception:
        pass
    return [text[:200]]


# ─────────────────────────────────────────────
#  WEB SEARCH — Serper
# ─────────────────────────────────────────────
def search_web(query: str) -> list:
    if not SERPER_API_KEY:
        return []
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5}, timeout=10
        )
        resp.raise_for_status()
        return [{"title": r.get("title",""), "snippet": r.get("snippet",""), "source": r.get("link","")}
                for r in resp.json().get("organic", [])[:4]]
    except Exception as e:
        st.error(f"Web search failed: {e}")
        return []


# ─────────────────────────────────────────────
#  GOOGLE FACT CHECK DB (optional)
# ─────────────────────────────────────────────
def check_factcheck_db(query: str) -> list:
    if not GOOGLE_FC_KEY:
        return []
    try:
        resp = requests.get(
            GFC_URL,
            params={"query": query[:200], "key": GOOGLE_FC_KEY, "languageCode": "en"},
            timeout=10
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("claims", [])[:3]:
            review = item.get("claimReview", [{}])[0]
            results.append({
                "claim":     item.get("text", ""),
                "rating":    review.get("textualRating", "unknown"),
                "publisher": review.get("publisher", {}).get("name", "unknown"),
                "url":       review.get("url", "")
            })
        return results
    except Exception:
        return []


# ─────────────────────────────────────────────
#  EVIDENCE SYNTHESIS — Groq
# ─────────────────────────────────────────────
def synthesise_verdict(original_text, style_pred, style_conf, claims, web_evidence, fc_results) -> dict:
    evidence_block = "".join(
        f"[Web {i}] {ev['title']}\n{ev['snippet']}\nSource: {ev['source']}\n\n"
        for i, ev in enumerate(web_evidence, 1)
    )
    fc_block   = "".join(f"Fact-check: '{fc['claim']}' rated '{fc['rating']}' by {fc['publisher']}\n" for fc in fc_results)
    claims_str = "\n".join(f"- {c}" for c in claims)

    prompt = f"""You are a senior fact-checker. Analyse the following and give a final verdict.

ORIGINAL NEWS TEXT (first 800 chars):
{original_text[:800]}

CLAIMS IDENTIFIED:
{claims_str}

STYLE-DETECTION MODEL RESULT:
Prediction: {style_pred}, Confidence: {style_conf:.1%}
(This model only detects writing style patterns — NOT actual facts)

WEB SEARCH EVIDENCE:
{evidence_block if evidence_block else "No web results available."}

EXISTING FACT-CHECKS:
{fc_block if fc_block else "No existing fact-checks found."}

TASK: Decide if the news is Real or Fake.
- Prioritise web evidence and fact-checks over the style model.
- If web evidence clearly contradicts the claims → Fake even if style says Real.
- If evidence is missing or inconclusive → trust style model.
- Give a clear reason (2-3 sentences max).

Return ONLY valid JSON (no markdown, no preamble):
{{"verdict": "Real or Fake", "confidence": 0.0 to 1.0, "reason": "2-3 sentence explanation citing specific evidence", "evidence_used": true or false}}"""

    raw = call_llm(prompt)
    if not raw:
        return {"verdict": style_pred, "confidence": style_conf,
                "reason": "LLM unavailable — falling back to style model.", "evidence_used": False}
    try:
        clean  = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)
        if all(k in result for k in ["verdict", "confidence", "reason"]):
            result["verdict"]    = "Real" if "real" in result["verdict"].lower() else "Fake"
            result["confidence"] = float(result["confidence"])
            return result
    except Exception:
        pass
    return {"verdict": style_pred, "confidence": style_conf,
            "reason": "Could not parse LLM response — falling back to style model.", "evidence_used": False}


# ─────────────────────────────────────────────
#  HELPERS — render result box
# ─────────────────────────────────────────────
def render_result_box(tag_label, tag_class, verdict, confidence, reason):
    box_class   = "result-box-real" if verdict == "Real" else "result-box-fake"
    label_class = "result-real"     if verdict == "Real" else "result-fake"
    icon        = "✓"               if verdict == "Real" else "✗"
    st.markdown(f"""
<div class="result-box {box_class}">
  <div><span class="result-tag {tag_class}">{tag_label}</span></div>
  <div class="result-label {label_class}">{icon} {verdict}</div>
  <div class="result-conf">Confidence: {confidence:.1%}</div>
  <div class="result-reason">{reason}</div>
</div>""", unsafe_allow_html=True)


def render_claims(claims):
    items = "".join(
        f'<div class="claim-item"><div class="claim-num">#{i}</div><div class="claim-text">{c}</div></div>'
        for i, c in enumerate(claims, 1)
    )
    st.markdown(f'<div class="claims-wrapper">{items}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="collapsed"
)
inject_css()

# ── Hero ──────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Truth<span>Lens</span></div>
  <div class="hero-subtitle">Multilingual Fake News Detection System &nbsp;·&nbsp; Minor Project &nbsp;·&nbsp; Deep Learning</div>
  <div class="hero-badges">
    <span class="badge badge-teal">XLM-RoBERTa</span>
    <span class="badge badge-blue">Llama 3.3 70B</span>
    <span class="badge badge-amber">Live Web Search</span>
    <span class="badge badge-teal">100+ Languages</span>
    <span class="badge badge-red">Image Detection — Coming Soon</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Key status bar ────────────────────────────
serper_ok = bool(SERPER_API_KEY)
groq_ok   = bool(GROQ_API_KEY)
gfc_ok    = bool(GOOGLE_FC_KEY)
st.markdown(f"""
<div class="key-status">
  <span class="{'key-ok' if serper_ok else 'key-fail'}">{'●' if serper_ok else '○'} Serper {'connected' if serper_ok else 'missing'}</span>
  <span class="{'key-ok' if groq_ok   else 'key-fail'}">{'●' if groq_ok   else '○'} Groq {'connected' if groq_ok else 'missing'}</span>
  <span class="{'key-ok' if gfc_ok    else 'key-opt'}" >{'●' if gfc_ok    else '○'} Fact-Check DB {'connected' if gfc_ok else 'optional'}</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────
tab_text, tab_image = st.tabs(["📰  Text Detection", "🖼️  Image Detection"])

# ═══════════════════════════════════════════════
#  TAB 1 — TEXT DETECTION
# ═══════════════════════════════════════════════
with tab_text:
    col_input, col_info = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown('<div class="section-header">News Input</div>', unsafe_allow_html=True)
        user_input = st.text_area(
            label="news_input", label_visibility="collapsed", height=200,
            placeholder="Paste your news article or headline here — English, Hindi, or any of 100+ languages supported..."
        )
        btn_col, tog_col = st.columns([3, 1])
        with btn_col:
            analyze_btn = st.button("🔍  Analyze News", use_container_width=True, type="primary")
        with tog_col:
            mode_fast = st.toggle("Style only", value=False,
                                  help="Skip web search — instant result from model only")

    with col_info:
        st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card">
  <div class="layer-row"><div class="dot dot-amber"></div><div class="layer-name">Style Detector</div>XLM-RoBERTa — writing patterns</div>
  <div class="layer-row"><div class="dot dot-indigo"></div><div class="layer-name">Claim Extractor</div>Llama 3.3 — pulls verifiable facts</div>
  <div class="layer-row"><div class="dot dot-amber"></div><div class="layer-name">Web Search</div>Serper — live news for each claim</div>
  <div class="layer-row"><div class="dot dot-indigo"></div><div class="layer-name">Fact-Check DB</div>Google — existing verdicts</div>
  <div class="layer-row"><div class="dot dot-teal"></div><div class="layer-name">LLM Synthesiser</div>Llama 3.3 — combines all signals</div>
</div>
""", unsafe_allow_html=True)

    # ── Load model ────────────────────────────
    try:
        model, tokenizer = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        model = tokenizer = None
        st.warning(f"⚠ Could not load model: {e}")

    # ─────────────────────────────────────────
    #  ANALYSIS PIPELINE
    # ─────────────────────────────────────────
    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter some news text before analyzing.")
            st.stop()

        st.markdown("---")

        # ── Layer 1: Style ────────────────────
        style_pred, style_conf, prob_real, prob_fake = "Unknown", 0.5, 0.5, 0.5
        if model_loaded:
            with st.spinner("Running XLM-RoBERTa style detection..."):
                style_pred, style_conf, prob_real, prob_fake = predict_article(user_input, model, tokenizer)

        # ── Layer 2: Fact verification ────────
        claims, all_evidence, fc_hits, final = [], [], [], None

        if mode_fast or not (SERPER_API_KEY and GROQ_API_KEY):
            final = {
                "verdict":        style_pred,
                "confidence":     style_conf,
                "reason":         "Style only mode — fact verification skipped." if mode_fast
                                  else "API keys missing — showing style result only.",
                "evidence_used":  False
            }
        else:
            with st.spinner("Extracting verifiable claims with Llama 3.3..."):
                claims = extract_claims(user_input)

            all_evidence = []
            with st.spinner(f"Searching web for {len(claims)} claim(s)..."):
                for claim in claims:
                    all_evidence.extend(search_web(claim))
                    time.sleep(0.3)

            if GOOGLE_FC_KEY:
                with st.spinner("Querying Google Fact-Check database..."):
                    fc_hits = check_factcheck_db(" ".join(claims[:2]))

            with st.spinner("Synthesising final verdict with Llama 3.3..."):
                final = synthesise_verdict(user_input, style_pred, style_conf, claims, all_evidence, fc_hits)

        # ══════════════════════════════════════
        #  DISPLAY — Style vs LLM side by side
        # ══════════════════════════════════════
        st.markdown('<div class="section-header">Results Comparison</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2, gap="large")

        with res_col1:
            render_result_box(
                "🧠 Style Model — XLM-RoBERTa", "tag-style",
                style_pred, style_conf,
                "Based on writing patterns, linguistic style, and tone of the article. Does not verify actual facts."
            )
            m1, m2 = st.columns(2)
            m1.metric("Real probability", f"{prob_real:.1%}")
            m2.metric("Fake probability", f"{prob_fake:.1%}")
            st.progress(prob_real)
            st.caption("⚠ Style only — writing pattern analysis, not fact verification")

        with res_col2:
            render_result_box(
                "🌐 LLM Verdict — Llama 3.3 + Web", "tag-llm",
                final["verdict"], final["confidence"],
                final["reason"]
            )
            ev_label = "Live web + fact-check DB" if final.get("evidence_used") else "Style model fallback"
            m3, m4 = st.columns(2)
            m3.metric("LLM Confidence", f"{final['confidence']:.1%}")
            m4.metric("Evidence", "Web ✓" if final.get("evidence_used") else "None")
            st.progress(final["confidence"])
            st.caption("✓ Fact-verified — searches web in real-time" if final.get("evidence_used") else "⚠ No web evidence used")

        # ── Claims section ────────────────────
        if claims:
            st.markdown("---")
            st.markdown('<div class="section-header">Verifiable Claims Extracted</div>', unsafe_allow_html=True)
            render_claims(claims)

        # ── Evidence section ──────────────────
        if all_evidence:
            with st.expander(f"📄  Web Evidence — {len(all_evidence)} results found"):
                ev_html = ""
                for ev in all_evidence:
                    src = f'<a href="{ev["source"]}" class="ev-source" target="_blank">↗ {ev["source"][:60]}</a>' if ev["source"] else ""
                    ev_html += f'<div class="ev-card"><div class="ev-title">{ev["title"]}</div><div class="ev-snippet">{ev["snippet"]}</div>{src}</div>'
                st.markdown(ev_html, unsafe_allow_html=True)

        if fc_hits:
            with st.expander(f"✅  Fact-Check Database — {len(fc_hits)} matches"):
                for fc in fc_hits:
                    st.markdown(f"**{fc['claim']}**")
                    st.markdown(f"`{fc['rating']}` — {fc['publisher']}")
                    if fc["url"]:
                        st.markdown(f"[Read full fact-check →]({fc['url']})")
                    st.divider()

        # ── Final Verdict banner ──────────────
        st.markdown("---")
        st.markdown('<div class="section-header">Final Verdict</div>', unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns([3, 1, 1])
        with fc1:
            cls  = "verdict-real" if final["verdict"] == "Real" else "verdict-fake"
            icon = "✓ Real News"  if final["verdict"] == "Real" else "✗ Fake News"
            color = "#00c8b4"     if final["verdict"] == "Real" else "#ef4444"
            st.markdown(f"""
<div class="result-box result-box-{'real' if final['verdict']=='Real' else 'fake'}">
  <div><span class="result-tag tag-final">FINAL VERDICT</span></div>
  <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{color};margin-bottom:0.4rem;">{icon}</div>
  <div class="result-reason">{final['reason']}</div>
</div>""", unsafe_allow_html=True)
        with fc2:
            st.metric("Confidence", f"{final['confidence']:.1%}")
        with fc3:
            st.metric("Verified by", "Live Web" if final.get("evidence_used") else "Style Model")

        if not final.get("evidence_used"):
            st.warning("⚠ Verdict based on writing style only. Add SERPER + GROQ keys for full fact verification.")
        else:
            st.success("✓ Verdict is backed by live web evidence and LLM synthesis.")


# ═══════════════════════════════════════════════
#  TAB 2 — IMAGE DETECTION (COMING SOON)
# ═══════════════════════════════════════════════
with tab_image:
    st.markdown("""
<div class="coming-soon">
  <div class="coming-soon-icon">🖼️</div>
  <div class="coming-soon-title">Image Deepfake Detection</div>
  <div class="coming-soon-sub">
    Our EfficientNetB0-based deepfake face detection model is trained and ready.
    Full integration into this interface is currently in progress.
  </div>
  <div class="coming-soon-badge">🚧 COMING SOON</div>

  <div class="planned-features">
    <div class="pf-item">
      <strong>EfficientNetB0</strong>
      Transfer learning on 5 deepfake datasets
    </div>
    <div class="pf-item">
      <strong>Face Detection</strong>
      Automated face region extraction & analysis
    </div>
    <div class="pf-item">
      <strong>Confidence Score</strong>
      Real vs Fake probability with heatmap overlay
    </div>
    <div class="pf-item">
      <strong>Multi-dataset Trained</strong>
      140K Faces · Celeb-DF v2 · CIPLAB · HardFake
    </div>
  </div>
</div>
""", unsafe_allow_html=True)