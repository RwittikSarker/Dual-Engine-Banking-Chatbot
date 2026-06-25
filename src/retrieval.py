"""
retrieval.py
------------
Retrieval-Augmented Generation (RAG) engine.

Steps:
1. Embed all training examples with MiniLM → FAISS index.
2. At query time: embed query, search FAISS, retrieve top-k.
3. Build context prompt → pass to Flan-T5 for answer generation.

Usage (build index):
    python src/retrieval.py --build
    python src/retrieval.py --build --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils import ensure_dir, get_logger, load_config, load_json, save_json

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

class MiniLMEmbedder:
    """Wraps sentence-transformers/all-MiniLM-L6-v2 for encoding."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, D)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ──────────────────────────────────────────────
# FAISS Index
# ──────────────────────────────────────────────

class FAISSRetriever:
    """Inner-product (cosine) FAISS index over Banking77 training examples."""

    def __init__(self):
        import faiss
        self._faiss = faiss
        self.index: Optional[object] = None
        self.metadata: list[dict] = []  # [{text, label_name}]

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Build the FAISS index from precomputed embeddings."""
        dim = embeddings.shape[1]
        self.index = self._faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata
        logger.info("FAISS index built with %d vectors (dim=%d)", self.index.ntotal, dim)

    def save(self, index_path: str, metadata_path: str) -> None:
        ensure_dir(Path(index_path).parent)
        self._faiss.write_index(self.index, index_path)
        save_json(self.metadata, metadata_path)
        logger.info("FAISS index saved → %s", index_path)

    def load(self, index_path: str, metadata_path: str) -> None:
        self.index = self._faiss.read_index(index_path)
        self.metadata = load_json(metadata_path)
        logger.info("FAISS index loaded: %d vectors", self.index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top-k results with keys: text, label_name, similarity."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(query, top_k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["similarity"] = float(sim)
            results.append(entry)
        return results


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────

class FlanT5Generator:
    """Wraps google/flan-t5-small for response generation."""

    def __init__(self, model_name: str = "google/flan-t5-small"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        logger.info("Loading generator: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info("Generator loaded on device: %s", self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_beams: int = 2,
        do_sample: bool = False,
    ) -> str:
        """
        Generate a response from the given prompt.

        Note: temperature is incompatible with beam search (num_beams > 1).
        Use do_sample=True with num_beams=1 if you want temperature-based sampling.
        This implementation uses beam search (do_sample=False) for quality.
        """
        import torch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,       # Bug fix: was missing, caused inconsistency
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ──────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────

class RAGPipeline:
    """Full Retrieval-Augmented Generation pipeline."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.embedder = MiniLMEmbedder(cfg["retrieval"]["embedding_model"])
        self.retriever = FAISSRetriever()
        self.generator = FlanT5Generator(cfg["generation"]["model_name"])

        index_path = cfg["paths"]["faiss_index"]
        metadata_path = cfg["paths"]["faiss_metadata"]

        if Path(index_path).exists():
            self.retriever.load(index_path, metadata_path)
        else:
            logger.warning("FAISS index not found. Run: python src/retrieval.py --build")

    def answer(
        self,
        query: str,
        detected_intent: str = "unknown",
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Run full RAG for a query.

        Returns:
            dict with keys: generated_answer, retrieval_similarity, retrieved_examples
        """
        k = top_k or self.cfg["retrieval"]["top_k"]
        gen_cfg = self.cfg["generation"]

        # Embed query
        q_emb = self.embedder.encode_single(query)

        # Retrieve
        results = self.retriever.search(q_emb, top_k=k)
        top_similarity = results[0]["similarity"] if results else 0.0

        # Build prompt
        context_lines = []
        for i, r in enumerate(results, 1):
            context_lines.append(f"Example {i}: '{r['text']}' → Intent: {r['label_name']}")
        context = "\n".join(context_lines)

        prompt = (
            f"You are a helpful banking assistant. "
            f"The customer's query is: \"{query}\"\n"
            f"The detected intent is: {detected_intent}\n\n"
            f"Similar customer queries and their intents:\n{context}\n\n"
            f"Provide a clear, helpful, and professional response to the customer's query."
        )

        # Generate (beam search, no temperature conflict)
        answer = self.generator.generate(
            prompt,
            max_new_tokens=gen_cfg["max_new_tokens"],
            num_beams=gen_cfg["num_beams"],
            do_sample=gen_cfg.get("do_sample", False),
        )

        return {
            "generated_answer": answer,
            "retrieval_similarity": round(top_similarity, 4),
            "retrieved_examples": results,
        }


# ──────────────────────────────────────────────
# Build index (CLI)
# ──────────────────────────────────────────────

def build_index(cfg: dict) -> None:
    """Embed all training examples and build the FAISS index."""
    data_dir = Path(cfg["paths"]["data_dir"])
    label_names: list[str] = load_json(data_dir / "label_names.json")

    rows = load_json(data_dir / "train.json")
    texts = [r["text"] for r in rows]
    label_ids = [r["label"] for r in rows]
    label_name_list = [label_names[lid] for lid in label_ids]

    embedder = MiniLMEmbedder(cfg["retrieval"]["embedding_model"])
    logger.info("Encoding %d training examples…", len(texts))
    embeddings = embedder.encode(texts, show_progress=True)

    metadata = [{"text": t, "label_name": l} for t, l in zip(texts, label_name_list)]

    retriever = FAISSRetriever()
    retriever.build(embeddings, metadata)
    retriever.save(cfg["paths"]["faiss_index"], cfg["paths"]["faiss_metadata"])
    logger.info("Index build complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval engine utilities")
    parser.add_argument("--build", action="store_true", help="Build FAISS index from training data")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.build:
        build_index(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
