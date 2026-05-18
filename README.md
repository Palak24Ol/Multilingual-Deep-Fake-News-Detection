# 🔍 Multilingual Deep Fake News Detection

A **multi-modal fake news detection system** that analyzes both **text articles** and **images** to classify news as Real or Fake. Supports **English and Hindi** with a 3-layer verification pipeline powered by XLM-RoBERTa and Llama 3.3.

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
│   ├── datasetfile.ipynb           # Dataset collection & merging
│   ├── model.ipynb                 # XLM-RoBERTa model training
│   ├── predict-model.ipynb         # Model evaluation & prediction
│   ├── internetnewsdataset.ipynb   # Internet news dataset integration
│   └── finetunedmodel.ipynb        # Fine-tuning with updated dataset
│
├── DeepFakeNewsDetection_Image/
│   └── deep-fake-image-final.ipynb # EfficientNetB0 deepfake image detection
│
├── Dataset/                        # Sample dataset files
├── app.py                          # Streamlit web application
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
| Languages | English, Hindi |
| Fine-tuned on | Multilingual fake news datasets (see below) |

### Image Detection
| Component | Detail |
|---|---|
| Model | `EfficientNetB0` (transfer learning) |
| Task | Deepfake face classification |
| Input Size | 128×128 |
| Datasets | 140K Faces, Celeb-DF v2, CIPLAB, HardFakeVsReal |

---

## 📊 Datasets Used

### Text Datasets (Kaggle)
| Dataset | Link |
|---|---|
| Dataset 1 | <!-- Add your Kaggle dataset link here --> |
| Dataset 2 | <!-- Add your Kaggle dataset link here --> |
| Internet News Dataset | <!-- Add your Kaggle dataset link here --> |

### Image Datasets (Kaggle)
| Dataset | Link |
|---|---|
| 140K Real and Fake Faces | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces |
| Deepfake and Real Images | https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images |
| Celeb-DF v2 | https://www.kaggle.com/datasets/reubensuju/celeb-df-v2 |
| Real and Fake Face Detection (CIPLAB) | https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection |
| HardFakeVsRealFaces | https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces |

---

## 🖥️ Streamlit UI

> Screenshots coming soon <!-- Replace with actual screenshots -->

The UI supports:
- Paste any news article (English or Hindi)
- Toggle between **Full Verification** (web search + LLM) and **Style Only** (fast mode)
- Displays: Verdict card, Confidence %, Identified Claims, Evidence sources

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Multilingual-Deep-Fake-News-Detection.git
cd Multilingual-Deep-Fake-News-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root:
```env
SERPER_API_KEY=your_serper_api_key
GROQ_API_KEY=your_groq_api_key
GOOGLE_FC_API_KEY=your_google_factcheck_key   # optional
```

### 4. Download model files
Download the fine-tuned models and place them in the root:
- `improved_model_v2.pt` → [Download Link](#) <!-- Add Google Drive link -->
- `xlm-roberta-base/` folder → [Download Link](#) <!-- Add Google Drive link -->

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

| Key | Where to get |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com |
| `SERPER_API_KEY` | https://serper.dev |
| `GOOGLE_FC_API_KEY` | https://developers.google.com/fact-check/tools/api (optional) |

---

## 📄 Project Report

The full project report is included in this repository → [View Report](./Project_Report.pdf)

---

## 🛠️ Tech Stack

- **ML/DL**: PyTorch, HuggingFace Transformers, TensorFlow, Keras
- **Models**: XLM-RoBERTa, EfficientNetB0, Llama 3.3 (via Groq)
- **APIs**: Groq, Serper, Google Fact Check
- **Frontend**: Streamlit
- **Languages**: Python

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [your_linkedin](#)

---

## 📜 License

This project is for academic/educational purposes only.