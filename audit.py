import io
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image

from models import load_clip_model, generate_advice

# Prompts used for simple CLIP-based scoring
_PROFESSIONAL_POS = [
    "professional headshot",
    "formal attire",
    "clean background",
]
_PROFESSIONAL_NEG = [
    "casual selfie",
    "party picture",
]
_APPROACHABLE_POS = ["friendly smile", "approachable expression"]
_APPROACHABLE_NEG = ["stern expression", "angry face"]


@lru_cache()
def _get_text_embeddings(prompts):
    processor, model = load_clip_model()
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        return model.get_text_features(**inputs)


def _score_clip(image_embedding, pos_emb, neg_emb) -> float:
    pos_score = F.cosine_similarity(image_embedding, pos_emb.mean(0, keepdim=True))
    neg_score = F.cosine_similarity(image_embedding, neg_emb.mean(0, keepdim=True))
    score = ((pos_score - neg_score) + 1) * 5  # map from [-1,1] to [0,10]
    return max(1.0, min(10.0, score.item()))


def analyze(image_data: bytes, platform: str) -> dict:
    """Analyze an uploaded profile image and return scores with advice."""
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    processor, model = load_clip_model()
    img_inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        image_emb = model.get_image_features(**img_inputs)

    prof_pos = _get_text_embeddings(tuple(_PROFESSIONAL_POS))
    prof_neg = _get_text_embeddings(tuple(_PROFESSIONAL_NEG))
    appr_pos = _get_text_embeddings(tuple(_APPROACHABLE_POS))
    appr_neg = _get_text_embeddings(tuple(_APPROACHABLE_NEG))

    professionalism = _score_clip(image_emb, prof_pos, prof_neg)
    approachability = _score_clip(image_emb, appr_pos, appr_neg)

    result = {
        "message": f"Received image of {len(image_data)} bytes for {platform}",
        "professionalism_score": int(round(professionalism)),
        "approachability_score": int(round(approachability)),
    }
    result["summary"] = generate_advice(result, platform)
    return result
