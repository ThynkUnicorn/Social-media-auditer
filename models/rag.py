from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from transformers import AutoTokenizer

from .loaders import load_text_generator


class FAQRetriever:
    """Simple embedding-based FAQ retriever."""

    def __init__(self, faq_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.faq_texts = self._load_faq(faq_path)
        self.embedder = SentenceTransformer(embed_model)
        self.embeddings = self.embedder.encode(self.faq_texts, convert_to_tensor=True)

    @staticmethod
    def _load_faq(path: str) -> List[str]:
        text = Path(path).read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def query(self, question: str, top_k: int = 1) -> List[str]:
        q_emb = self.embedder.encode([question], convert_to_tensor=True)
        scores = F.cosine_similarity(q_emb, self.embeddings)[0]
        top_indices = torch.topk(scores, k=top_k).indices
        return [self.faq_texts[i] for i in top_indices]


def generate_advice(result: dict, platform: str, faq_path: str = "faq/faq.txt") -> str:
    """Generate advice using a simple RAG pipeline."""
    retriever = FAQRetriever(faq_path)
    relevant = "\n".join(retriever.query(f"tips for {platform} profile picture"))
    tokenizer, model = load_text_generator()

    prompt = (
        f"You are an assistant helping with social media profile photos. "
        f"Here are analysis results: {result}.\n"
        f"Relevant guideline: {relevant}\n"
        f"Provide a short suggestion.""
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
