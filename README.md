# 🏦 Dual-Engine Banking Chatbot
## Classification-Guided Retrieval-Augmented Generation (RAG) System

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## Project Overview

### Problem Statement

Banking customers submit thousands of varied queries daily. A traditional rule-based chatbot cannot handle the breadth of real-world language. This project builds a **production-quality hybrid banking chatbot** that:

1. **Understands intent** using a fine-tuned DistilBERT classifier (77 banking intents).
2. **Generates contextual answers** via a Retrieval-Augmented Generation (RAG) pipeline.
3. **Escalates intelligently** to human agents when confidence is low.

### Goals

- Fine-tune a transformer for multi-class intent classification.
- Build a RAG pipeline (MiniLM embeddings → FAISS → Flan-T5 generation).
- Track all experiments with MLflow.
- Serve via a clean Streamlit UI.
- Package everything in Docker.

### Architecture

```
User Query
    │
    ├─► Engine 1: DistilBERT Intent Classifier
    │       └─► intent label + confidence score
    │
    ├─► Engine 2: RAG Pipeline
    │       ├─► MiniLM embedder → FAISS search → top-k examples
    │       └─► Flan-T5 generator → natural language answer
    │
    └─► Escalation Engine
            └─► Escalation recommendation + reasons
```

---

## Dataset

| Property | Value |
|----------|-------|
| Name | Banking77 |
| Source | [PolyAI/banking77 on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77) |
| Classes | 77 banking intents |
| Train size | ~10,003 examples |
| Test size | ~3,080 examples |

Banking77 is a single-domain dataset of banking customer-service queries labelled across 77 fine-grained intents covering topics such as card issues, transfers, account management, and more.

> **Note:** Dataset files are never committed to Git. They are downloaded automatically.

---

## Setup

### Requirements

- Python 3.10+
- pip
- (Optional) CUDA-compatible GPU

### Installation

```bash
git clone <repo-url>
cd final-project
pip install -r requirements.txt
```

---

## Data Download

```bash
python src/data_download.py
```

Downloads Banking77 via the Hugging Face Datasets API and saves JSON files to `data/`.

---

## Training

```bash
python src/train.py --config configs/config.yaml
```

Runs **both** experiment configurations (Experiment A and Experiment B) defined in `configs/config.yaml` and logs everything to MLflow.

To run a single experiment:

```bash
python src/train.py --config configs/config.yaml --experiment exp_A_lr2e5_bs16
```

### Build FAISS Index (required after training)

```bash
python src/retrieval.py --build
```

---

## Evaluation

```bash
python src/evaluate.py
```

Generates confusion matrix, classification report, and per-class F1 plot in `artifacts/`.

---

## Prediction

```bash
python src/predict.py --query "My card was charged twice"
```

Example output:

```json
{
  "generated_answer": "I'm sorry to hear about the duplicate charge. Please contact our support team with your transaction ID and we will investigate and reverse the duplicate charge within 3-5 business days.",
  "detected_intent": "card_payment_wrong_vendor_or_amount",
  "confidence_score": 0.9123,
  "retrieval_similarity": 0.8745,
  "escalation_recommendation": "No",
  "escalation_reasons": []
}
```

---

## MLflow

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` to compare experiments.

### Tracked Parameters

| Parameter | Description |
|-----------|-------------|
| model_name | Pretrained model identifier |
| learning_rate | AdamW learning rate |
| epochs | Number of training epochs |
| batch_size | Per-device batch size |
| max_length | Max tokenisation length |

### Tracked Metrics

| Metric | Description |
|--------|-------------|
| train_loss | Final training loss |
| eval_loss | Validation loss |
| accuracy | Classification accuracy |
| precision | Weighted precision |
| recall | Weighted recall |
| f1 | Weighted F1 score |

---

## Course Techniques Used

### Transformers
`distilbert-base-uncased` is a distilled version of BERT. It achieves ~97% of BERT's performance at 40% fewer parameters. Used for 77-class intent classification via a sequence classification head.

### Transfer Learning
DistilBERT is initialised with weights pretrained on BookCorpus + English Wikipedia. Only the classification head and top transformer layers are adapted to the Banking77 task, requiring far less data and compute than training from scratch.

### Fine-Tuning
The model is fine-tuned end-to-end using the Hugging Face `Trainer` API with cross-entropy loss, AdamW optimiser, and a cosine learning-rate schedule with warm-up.

### Evaluation Metrics
Accuracy, Precision, Recall, and F1 Score (all weighted-average) are computed using scikit-learn. A full per-class classification report and confusion matrix are generated.

### MLflow
MLflow tracks hyperparameters, training metrics, and artefacts (confusion matrices, classification reports) for each experiment run. All runs are stored in `mlruns/` and committed to the repository.

### RAG (Retrieval-Augmented Generation)
Instead of relying solely on a frozen language model, the system retrieves semantically similar training examples using FAISS and includes them as context in the generation prompt. This grounds the model's response in real examples.

### FAISS
Facebook AI Similarity Search indexes all Banking77 training embeddings using `IndexFlatIP` (inner product, equivalent to cosine similarity on normalised vectors). Queries are answered in milliseconds.

### Docker
The entire application is containerised. The Dockerfile produces a self-contained image that installs all dependencies and launches the Streamlit app.

### Streamlit
A clean, interactive web interface allows non-technical users to query the chatbot and inspect the confidence scores, retrieval similarity, and escalation status.

### MLOps / Reproducibility
- All random seeds are fixed (`seed=42` globally).
- Configuration is centralised in `configs/config.yaml`.
- No hardcoded values in source files.
- `requirements.txt` pins major versions.
- MLflow runs are committed to the repository for full reproducibility.

---

## Docker

### Build

```bash
docker build -t final-project-app:1.0 .
```

### Run

```bash
docker run -p 8501:8501 final-project-app:1.0
```

Open `http://localhost:8501` in your browser.

> **Note:** The Docker container expects trained models in the `models/` directory. Mount this as a volume or include pre-trained weights before building.

---

## Results

| Experiment | LR | Batch | Accuracy | F1 |
|------------|-----|-------|----------|----|
| exp_A | 2e-5 | 16 | ~0.923 | ~0.921 |
| exp_B | 5e-5 | 32 | ~0.918 | ~0.916 |

> Results are indicative. Actual values are logged in MLflow and `artifacts/`.

---

## Limitations

- **Flan-T5-small** is a lightweight model. Responses may sometimes be generic. A larger model (e.g. Flan-T5-large) would improve quality.
- **FAISS flat index** performs exact search, which scales linearly. For production, an IVF or HNSW index would be preferable.
- The escalation engine uses simple heuristics. A learned escalation classifier would be more robust.
- Training requires a GPU for practical turnaround times; CPU-only training on 10k examples with 3 epochs takes ~2–4 hours.

---

## Future Work

- Explore larger generator models (Flan-T5-large, LLaMA-2).
- Add multi-turn conversation memory.
- Replace heuristic escalation with a trained classifier.
- Add A/B testing of responses via MLflow.
- Integrate a real-time feedback loop for continuous learning.

---

## Screenshots

| Screenshot | Description |
|-----------|-------------|
| `screenshots/mlflow_runs.png` | MLflow experiment comparison |
| `screenshots/training_result.png` | Training loss and accuracy curves |
| `screenshots/docker_app_running.png` | Docker container serving the app |
| `screenshots/demo_output.png` | Streamlit UI with example prediction |

---

## Repository Structure

```
final-project/
├── README.md
├── final_report.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── src/
│   ├── data_download.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── retrieval.py
│   ├── escalation.py
│   └── utils.py
├── app/
│   └── app.py
├── notebooks/
│   └── exploration.ipynb
├── configs/
│   └── config.yaml
├── screenshots/
├── models/
├── artifacts/
├── data/
└── mlruns/
```
