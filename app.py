import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import requests
import json
import os
import time
from dotenv import load_dotenv

# ─────────────────────────────────────────────
#  LOAD ENVIRONMENT VARIABLES FROM .env FILE
# ─────────────────────────────────────────────
load_dotenv()

SERPER_API_KEY  = os.getenv("SERPER_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GOOGLE_FC_KEY   = os.getenv("GOOGLE_FC_API_KEY", "")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH      = "C:/FakeNewsApp/improved_model_v2.pt"
BASE_MODEL_PATH = "./xlm-roberta-base"
MAX_LENGTH      = 256
STRIDE          = 128
MAX_CHUNKS      = 5
DEVICE          = torch.device("cpu")

GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"
SERPER_URL    = "https://google.serper.dev/search"
GFC_URL       = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


# ─────────────────────────────────────────────
#  CUSTOM CSS — clean light mode
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    .stApp { background-color: #f5f5f3 !important; }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 3rem 2rem 4rem 2rem !important;
        max-width: 720px !important;
        margin: 0 auto !important;
    }

    /* ── App title ── */
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111111;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.82rem;
        color: #888888;
        margin-bottom: 2rem;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #aaaaaa;
        margin-bottom: 0.6rem;
    }

    /* ── Textarea ── */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        color: #111111 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.92rem !important;
        padding: 0.9rem 1rem !important;
        box-shadow: none !important;
        transition: border-color 0.15s;
    }
    .stTextArea textarea:focus {
        border-color: #aaaaaa !important;
        box-shadow: none !important;
    }
    .stTextArea textarea::placeholder { color: #bbbbbb !important; }
    .stTextArea label { display: none !important; }

    /* ── Button ── */
    .stButton > button {
        background: #111111 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.65rem 1.5rem !important;
        height: 2.8rem !important;
        transition: opacity 0.15s !important;
        box-shadow: none !important;
        width: 100% !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    .stButton > button:active { opacity: 0.7 !important; }

    /* ── Verdict cards ── */
    .verdict-real {
        background: #f0faf4;
        border: 1px solid #b6e8c8;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
    }
    .verdict-fake {
        background: #fff4f4;
        border: 1px solid #f5c6c6;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
    }
    .verdict-tag {
        font-size: 1.4rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .verdict-real .verdict-tag  { color: #1a7a3f; }
    .verdict-fake .verdict-tag  { color: #c0392b; }
    .verdict-conf {
        font-size: 0.82rem;
        font-weight: 500;
    }
    .verdict-real .verdict-conf { color: #3aaa6a; }
    .verdict-fake .verdict-conf { color: #e05050; }

    /* ── Claim pills ── */
    .claim-pill {
        display: inline-block;
        background: #f0f0ee;
        border: 1px solid #e0e0de;
        color: #555555;
        font-size: 0.8rem;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem 0.25rem 0.25rem 0;
        line-height: 1.5;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #111111 !important; }

    /* ── Divider ── */
    hr { border-color: #e8e8e8 !important; margin: 1.5rem 0 !important; }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        font-size: 0.85rem !important;
        background: #fafafa !important;
    }

    /* ── Toggle ── */
    .stCheckbox label, .stToggle label {
        font-size: 0.85rem !important;
        color: #666666 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class HierarchicalXLMRBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_MODEL_PATH)
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
        cls_embeddings = outputs.last_hidden_state[:, 0, :].view(batch_size, num_chunks, -1)
        doc_outputs    = self.doc_transformer(cls_embeddings)
        attn_weights   = torch.softmax(self.attention(doc_outputs), dim=1)
        doc_rep        = torch.sum(attn_weights * doc_outputs, dim=1)
        return self.classifier(doc_rep)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model     = HierarchicalXLMRBase()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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
    input_ids      = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    if input_ids.size(0) > MAX_CHUNKS:
        input_ids, attention_mask = input_ids[:MAX_CHUNKS], attention_mask[:MAX_CHUNKS]

    pad = MAX_CHUNKS - input_ids.size(0)
    if pad > 0:
        input_ids      = torch.cat([input_ids,      torch.zeros((pad, MAX_LENGTH), dtype=torch.long)], dim=0)
        attention_mask = torch.cat([attention_mask, torch.zeros((pad, MAX_LENGTH), dtype=torch.long)], dim=0)

    with torch.no_grad():
        probs = torch.softmax(
            model(input_ids.unsqueeze(0).to(DEVICE), attention_mask.unsqueeze(0).to(DEVICE)),
            dim=1
        )
    prob_real, prob_fake = probs[0][0].item(), probs[0][1].item()
    prediction = "Real" if prob_real > prob_fake else "Fake"
    return prediction, max(prob_real, prob_fake), prob_real, prob_fake


# ─────────────────────────────────────────────
#  LAYER 2 — GROQ LLM HELPER
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


def extract_claims(text: str) -> list:
    prompt = f"""You are a fact-checking assistant. Extract 2 to 4 short, specific, verifiable factual claims from the news text below. These must be concrete facts (names, numbers, events, dates) that can be verified via a web search.

Return ONLY a valid JSON array of strings. No explanation, no markdown, no preamble.
Example: ["India won the T20 World Cup 2024", "Rohit Sharma scored 76 runs"]

News text:
{text[:1500]}"""
    raw = call_llm(prompt)
    if not raw:
        return [text[:200]]
    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        claims = json.loads(clean)
        if isinstance(claims, list) and claims:
            return [str(c) for c in claims[:4]]
    except Exception:
        pass
    return [text[:200]]


# ─────────────────────────────────────────────
#  LAYER 2 — WEB SEARCH (Serper)
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
#  LAYER 2 — GOOGLE FACT CHECK (optional)
# ─────────────────────────────────────────────
def check_factcheck_db(query: str) -> list:
    if not GOOGLE_FC_KEY:
        return []
    try:
        resp = requests.get(GFC_URL, params={"query": query[:200], "key": GOOGLE_FC_KEY, "languageCode": "en"}, timeout=10)
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
#  LAYER 3 — EVIDENCE SYNTHESIS (Groq)
# ─────────────────────────────────────────────
def synthesise_verdict(original_text, style_pred, style_conf, claims, web_evidence, fc_results) -> dict:
    evidence_block = "".join(f"[Web {i}] {ev['title']}\n{ev['snippet']}\nSource: {ev['source']}\n\n"
                             for i, ev in enumerate(web_evidence, 1))
    fc_block = "".join(f"Fact-check: '{fc['claim']}' rated '{fc['rating']}' by {fc['publisher']}\n"
                       for fc in fc_results)
    claims_str = "\n".join(f"- {c}" for c in claims)

    prompt = f"""You are a senior fact-checker. Analyse the following and give a final verdict.

ORIGINAL NEWS TEXT (first 800 chars):
{original_text[:800]}

CLAIMS IDENTIFIED:
{claims_str}

STYLE-DETECTION MODEL RESULT:
Prediction: {style_pred}, Confidence: {style_conf:.1%}
(This model only detects writing style patterns, NOT facts)

WEB SEARCH EVIDENCE:
{evidence_block if evidence_block else "No web results available."}

EXISTING FACT-CHECKS:
{fc_block if fc_block else "No existing fact-checks found."}

TASK: Decide if the news is Real or Fake.
- Prioritise web evidence and fact-checks over the style-detection result.
- If web evidence clearly contradicts the claims, mark as Fake even if style model says Real.
- If evidence is missing or inconclusive, trust the style model result.
- Give a short, clear reason (1-2 sentences max).

Return ONLY valid JSON (no markdown, no preamble):
{{"verdict": "Real or Fake", "confidence": 0.0 to 1.0, "reason": "one to two sentences", "evidence_used": true or false}}"""

    raw = call_llm(prompt)
    if not raw:
        return {"verdict": style_pred, "confidence": style_conf,
                "reason": "LLM synthesis unavailable — falling back to style model result.", "evidence_used": False}
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
#  STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

inject_css()

# ── Header ───────────────────────────────────
st.markdown('<div class="app-title">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">English & Hindi supported · Powered by XLM-RoBERTa + Llama 3.3</div>', unsafe_allow_html=True)

# ── Load Model ────────────────────────────────
try:
    model, tokenizer = load_model()
    model_loaded = True
except Exception:
    model_loaded = False
    model = tokenizer = None

# ── Input ─────────────────────────────────────
st.markdown('<div class="section-label">News Input</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="news_input",
    label_visibility="collapsed",
    height=160,
    placeholder="Paste your news article or headline here — English or Hindi supported..."
)

col_btn, col_toggle = st.columns([3, 1], gap="small")
with col_btn:
    analyze_btn = st.button("🔍  Analyze News", use_container_width=True)
with col_toggle:
    mode_fast = st.toggle("Style only", value=False, help="Skip web search — faster but no fact verification")


# ─────────────────────────────────────────────
#  ANALYSIS PIPELINE
# ─────────────────────────────────────────────
if analyze_btn:
    if not user_input.strip():
        st.warning("Please paste some news text before analyzing.")
        st.stop()

    st.markdown("---")

    # ── Layer 1: Style detection ──────────────
    if model_loaded:
        with st.spinner("Running style analysis..."):
            style_pred, style_conf, prob_real, prob_fake = predict_article(user_input, model, tokenizer)
    else:
        style_pred, style_conf = "Unknown", 0.5

    # ── Layer 2: Fact verification ────────────
    if mode_fast or not model_loaded:
        final = {
            "verdict":       style_pred,
            "confidence":    style_conf,
            "reason":        "Style only mode — web search skipped." if mode_fast else "Model not loaded.",
            "evidence_used": False
        }
        claims = extract_claims(user_input) if GROQ_API_KEY else []

    elif not SERPER_API_KEY or not GROQ_API_KEY:
        missing = []
        if not SERPER_API_KEY: missing.append("SERPER_API_KEY")
        if not GROQ_API_KEY:   missing.append("GROQ_API_KEY")
        st.error(f"Missing API keys in .env: {', '.join(missing)}")
        final = {"verdict": style_pred, "confidence": style_conf,
                 "reason": f"Missing keys: {', '.join(missing)}.", "evidence_used": False}
        claims = []

    else:
        with st.spinner("Extracting claims..."):
            claims = extract_claims(user_input)

        all_evidence = []
        with st.spinner(f"Searching web for {len(claims)} claim(s)..."):
            for claim in claims:
                all_evidence.extend(search_web(claim))
                time.sleep(0.3)

        fc_hits = []
        if GOOGLE_FC_KEY:
            with st.spinner("Querying fact-check database..."):
                fc_hits = check_factcheck_db(" ".join(claims[:2]))

        with st.spinner("Synthesising verdict..."):
            final = synthesise_verdict(user_input, style_pred, style_conf, claims, all_evidence, fc_hits)

    # ── Verdict card ──────────────────────────
    is_real   = final["verdict"] == "Real"
    card_cls  = "verdict-real" if is_real else "verdict-fake"
    tag_icon  = "✓ Real" if is_real else "✗ Fake"
    conf_pct  = f"{final['confidence']:.0%} confidence"

    st.markdown(f"""
<div class="{card_cls}">
  <div class="verdict-tag">{tag_icon}</div>
  <div class="verdict-conf">{conf_pct}</div>
</div>
""", unsafe_allow_html=True)

    # ── Claims ────────────────────────────────
    if claims:
        st.markdown('<div class="section-label">Claims identified</div>', unsafe_allow_html=True)
        pills_html = "".join(f'<span class="claim-pill">{c}</span>' for c in claims)
        st.markdown(f'<div style="margin-top:0.2rem">{pills_html}</div>', unsafe_allow_html=True)