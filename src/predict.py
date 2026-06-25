"""
predict.py
----------
Full dual-engine prediction pipeline.

Loads:
  - DistilBERT classifier
  - MiniLM embedder + FAISS retriever
  - Flan-T5 generator
  - Escalation engine

Returns a JSON prediction including intent, confidence, retrieval similarity,
generated answer, and escalation recommendation.

Usage:
    python src/predict.py --query "My card was charged twice"
    python src/predict.py --query "..." --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bug fix: always resolve paths relative to this file, not CWD
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from src.escalation import build_escalation_engine
from src.retrieval import MiniLMEmbedder, FAISSRetriever, FlanT5Generator
from src.utils import get_logger, load_config, load_label_encoder

logger = get_logger(__name__)


class DualEnginePipeline:
    """Combines the classifier, RAG retriever/generator, and escalation engine."""

    def __init__(self, cfg: dict):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg

        # ── Classifier ──────────────────────────────
        classifier_dir = Path(cfg["paths"]["classifier_dir"]) / "best"
        if not classifier_dir.exists():
            raise FileNotFoundError(
                f"Classifier not found at {classifier_dir}. Run train.py first."
            )
        logger.info("Loading classifier from %s", classifier_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(classifier_dir))
        self.classifier = AutoModelForSequenceClassification.from_pretrained(str(classifier_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        self.classifier.eval()
        logger.info("Classifier loaded on device: %s", self.device)

        # Label encoder
        self.label2id, self.id2label = load_label_encoder(Path(cfg["paths"]["label_encoder"]))

        # ── Retriever ────────────────────────────────
        self.embedder = MiniLMEmbedder(cfg["retrieval"]["embedding_model"])
        self.retriever = FAISSRetriever()
        index_path = cfg["paths"]["faiss_index"]
        metadata_path = cfg["paths"]["faiss_metadata"]
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Run: python src/retrieval.py --build"
            )
        self.retriever.load(index_path, metadata_path)

        # ── Generator ────────────────────────────────
        self.generator = FlanT5Generator(cfg["generation"]["model_name"])

        # ── Escalation ───────────────────────────────
        self.escalation_engine = build_escalation_engine(cfg)

    def classify(self, query: str) -> tuple[str, float]:
        """Return (intent_name, confidence_score)."""
        enc = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=self.cfg["classifier"]["max_length"],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.classifier(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())
        intent = self.id2label[top_idx]
        return intent, confidence

    def predict(self, query: str) -> dict:
        """Full pipeline: classify → retrieve → generate → escalate."""
        gen_cfg = self.cfg["generation"]

        # 1. Intent classification
        intent, confidence = self.classify(query)
        logger.info("Classified: %s (confidence=%.4f)", intent, confidence)

        # 2. Retrieval
        q_emb = self.embedder.encode_single(query)
        retrieved = self.retriever.search(q_emb, top_k=self.cfg["retrieval"]["top_k"])
        top_similarity = retrieved[0]["similarity"] if retrieved else 0.0

        # 3. Build prompt and generate
        context_lines = [
            f"Example {i}: '{r['text']}' → Intent: {r['label_name']}"
            for i, r in enumerate(retrieved, 1)
        ]
        context = "\n".join(context_lines)
        prompt = (
            f"You are a helpful banking customer support assistant.\n"
            f"Customer query: \"{query}\"\n"
            f"Detected intent: {intent}\n\n"
            f"Similar resolved queries:\n{context}\n\n"
            f"Provide a clear, empathetic, and actionable response:"
        )
        answer = self.generator.generate(
            prompt,
            max_new_tokens=gen_cfg["max_new_tokens"],
            num_beams=gen_cfg["num_beams"],
            do_sample=gen_cfg.get("do_sample", False),
        )

        # 4. Escalation
        esc_result = self.escalation_engine.evaluate(
            query=query,
            confidence_score=confidence,
            retrieval_similarity=top_similarity,
            generated_answer=answer,
        )

        return {
            "generated_answer": answer,
            "detected_intent": intent,
            "confidence_score": round(confidence, 4),
            "retrieval_similarity": round(top_similarity, 4),
            "escalation_recommendation": esc_result.recommendation,
            "escalation_reasons": esc_result.reasons,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking chatbot prediction pipeline")
    parser.add_argument("--query", required=True, help="User query text")
    # Bug fix: default config path relative to repo root, not CWD
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "config.yaml"),
        help="Path to config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        pipeline = DualEnginePipeline(cfg)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    result = pipeline.predict(args.query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
