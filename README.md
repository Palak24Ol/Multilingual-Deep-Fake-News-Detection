# 🔍 Multilingual Deep Fake News Detection

A **multi-modal fake news detection system** that analyzes both **text articles** and **images** to classify news as Real or Fake. Supports **100+ languages** (fine-tuned on English, Hindi, Bengali, Gujarati, Marathi & Telugu) with a 3-layer verification pipeline powered by XLM-RoBERTa and Llama 3.3.

## 🚀 Live Demo

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/PalakJaiswal2401/multilingual-fake-news-detection)

---

## 📌 Project Overview

This project was developed as a **Minor Project** for detecting misinformation using deep learning. It combines:
- **Text Analysis** — Multilingual fake news detection using a fine-tuned XLM-RoBERTa model
- **Image Analysis** — Deepfake face detection using EfficientNetB0
- **Fact Verification** — 3-layer pipeline with web search + LLM synthesis

---

## 🏗️ System Architecture

```
Input (Text / Image)
        │
        ▼
┌──────────────────────────────────────────┐
│         Layer 1 — Style Detection        │
│   XLM-RoBERTa (fine-tuned, multilingual) │
│   Hierarchical Transformer Architecture  │
└──────────────────────┬───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│       Layer 2 — Fact Verification        │
│  • Claim Extraction (Groq / Llama 3.3)   │
│  • Web Search (Serper API)               │
│  • Google Fact Check API                 │
└──────────────────────┬───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│      Layer 3 — Evidence Synthesis        │
│   LLM-based verdict with web evidence    │
└──────────────────────┬───────────────────┘
                       │
                       ▼
              Final Verdict (Real / Fake)
```

---

## 📁 Repository Structure

```
├── DeepFakeNewsDetection_Text/
│   ├── datasetfile.ipynb           # File A — Dataset collection & merging (10+ sources)
│   ├── model.ipynb                 # File B — XLM-RoBERTa model training
│   ├── predict-model.ipynb         # File C — Model evaluation & prediction
│   ├── internetnewsdataset.ipynb   # File D — Internet news dataset integration
│   └── finetunedmodel.ipynb        # File E — Fine-tuning with updated dataset
│
├── DeepFakeNewsDetection_Image/
│   └── deep-fake-image-final.ipynb # EfficientNetB0 deepfake image detection
│
├── Dataset/                        # Sample dataset files
├── app.py                          # Streamlit web application (main)
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🧠 Models Used

### Text Detection
| Component | Detail |
|---|---|
| Base Model | `XLM-RoBERTa-base` |
| Architecture | Hierarchical Transformer (chunk-level + doc-level) |
| Languages | English, Hindi, Bengali, Gujarati, Marathi, Telugu |
| Fine-tuned on | 10+ multilingual fake news datasets |

### Image Detection
| Component | Detail |
|---|---|
| Model | `EfficientNetB0` (transfer learning) |
| Task | Deepfake face classification |
| Input Size | 128×128 |
| Training | Two-phase (frozen base → full fine-tune) |

---

## 📊 Datasets Used

### 📝 File A — Text Dataset Collection (`datasetfile.ipynb`)

| # | Dataset | Language | Link |
|---|---|---|---|
| 1 | WELFake — Fake News Classification | English | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| 2 | Fake News Classification | English | [Kaggle](https://www.kaggle.com/datasets/aadyasingh55/fake-news-classification) |
| 3 | English-Hindi Fake News | English + Hindi | [Kaggle](https://www.kaggle.com/datasets/maanavghai/english-hindi-fake-news) |
| 4 | Hindi Fake News Dataset | Hindi | [Kaggle](https://www.kaggle.com/datasets/sudhanshukumar344/hindi-fake-news-dataset) |
| 5 | Fake & Real News | English | [Kaggle](https://www.kaggle.com/datasets/imbikramsaha/fake-real-news) |
| 6 | Gujarati / Marathi / Telugu Fake News | Gujarati, Marathi, Telugu | [Zenodo](https://zenodo.org/records/11408513) |
| 7 | Bangla Fake News | Bengali | [Kaggle](https://www.kaggle.com/datasets/hrithikmajumdar/bangla-fake-news) |
| 8 | ISOT Fake and Real News | English | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |

### 🌐 File D — Internet News Dataset (`internetnewsdataset.ipynb`)

| # | Dataset | Source | Link |
|---|---|---|---|
| Archive 10 | Indian News Articles | Indian News | [Kaggle](https://www.kaggle.com/datasets/jatinkalra17/indian-news-articles-dataset) |
| Archive 11 | BBC News Articles | BBC | [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/bbc-news-articles) |
| Archive 12 | Inshorts English Dataset | Inshorts | [Kaggle](https://www.kaggle.com/datasets/shivamtaneja2304/inshorts-dataset-english) |
| Archive 13 | News Articles Classification | Mixed | [Kaggle](https://www.kaggle.com/datasets/banuprakashv/news-articles-classification-dataset-for-nlp-and-ml) |
| Archive 14 | Times of India Headlines | Times of India | [Kaggle](https://www.kaggle.com/datasets/thedevastator/times-of-india-headlines-analysis) |
| Archive 15 | India Headlines 2001–2023 | India News | [Kaggle](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset) |
| Archive 16 | News Category Dataset | HuffPost | [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) |
| LIAR | LIAR Fake News Dataset | Political | [Kaggle](https://www.kaggle.com/datasets/csmalarkodi/liar-fake-news-dataset) |

### 🖼️ Image Datasets

| Dataset | Link |
|---|---|
| 140K Real and Fake Faces | [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |
| Deepfake and Real Images | [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) |
| Celeb-DF v2 | [Kaggle](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) |
| Real and Fake Face Detection (CIPLAB) | [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) |
| HardFakeVsRealFaces | [Kaggle](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces) |

---

## 🖥️ Streamlit UI

<img width="1565" height="813" alt="image" src="https://github.com/user-attachments/assets/ca1b1d2e-b126-4bad-bffa-0574f33f88fc" />

The UI supports:
- Paste any news article in **many languages**
- Toggle between **Full Verification** (web search + LLM) and **Style Only** (fast mode)
- Displays: Verdict card, Confidence %, Identified Claims, Evidence sources

---

## ⬇️ Model Download

The fine-tuned model is too large for GitHub. Download and place in the project root:

| File | Download |
|---|---|
| `improved_model_v2.pt` | [Google Drive](https://drive.google.com/file/d/1UxSxjeUEse1CUTFOj1nUoQL90b3bTXZI/view?usp=sharing) |
| `improved_model_v2.pt` | [HuggingFace Hub](https://huggingface.co/PalakJaiswal2401/multilingual-fake-news-detection) |
| `xlm-roberta-base/` folder | Auto-download via HuggingFace (see below) |

```python
# Auto-download xlm-roberta-base
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Palak24Ol/Multilingual-Deep-Fake-News-Detection.git
cd Multilingual-Deep-Fake-News-Detection
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root:
```env
SERPER_API_KEY=your_serper_api_key
GROQ_API_KEY=your_groq_api_key
GOOGLE_FC_API_KEY=your_google_factcheck_key   # optional
```

### 4. Download the fine-tuned model
Download `improved_model_v2.pt` from [HuggingFace](https://huggingface.co/PalakJaiswal2401/multilingual-fake-news-detection) or [Google Drive](https://drive.google.com/file/d/1UxSxjeUEse1CUTFOj1nUoQL90b3bTXZI/view?usp=sharing) and place it in the project root.

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

| Key | Free? | Where to get |
|---|---|---|
| `GROQ_API_KEY` | ✅ Free | [console.groq.com](https://console.groq.com) |
| `SERPER_API_KEY` | ✅ Free tier | [serper.dev](https://serper.dev) |
| `GOOGLE_FC_API_KEY` | ✅ Free | [Google Fact Check API](https://developers.google.com/fact-check/tools/api) |

---

## 📄 Project Report

The full project report is included → [View Report](https://drive.google.com/file/d/1DdrmTtYAblPX9T6QKfVW3zZjJgNTHlMj/view?usp=sharing)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| ML / NLP | PyTorch, HuggingFace Transformers, XLM-RoBERTa |
| Image DL | TensorFlow, Keras, EfficientNetB0 |
| LLM | Llama 3.3 70B via Groq API |
| APIs | Serper (web search), Google Fact Check |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## 👩‍💻 Author

**Palak Jaiswal**
- GitHub: [@Palak24Ol](https://github.com/Palak24Ol)

---

## 📜 License

This project is for academic and educational purposes only.