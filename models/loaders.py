from functools import lru_cache
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    ViTModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache()
def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """Load CLIP model and processor."""
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(get_device())
    return processor, model


@lru_cache()
def load_vit_model(model_name: str = "google/vit-base-patch16-224"):
    """Load ViT model and processor."""
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(get_device())
    return processor, model


@lru_cache()
def load_text_generator(model_name: str = "google/flan-t5-base"):
    """Load text generation model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(get_device())
    return tokenizer, model
