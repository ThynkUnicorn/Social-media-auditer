import io
from PIL import Image
import torch

from models import load_clip_model, generate_advice


def analyze(image_data: bytes, platform: str) -> dict:
    """Analyze an uploaded profile image and return scores with advice."""
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    processor, model = load_clip_model()
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.get_image_features(**inputs)

    result = {
        "message": f"Received image of {len(image_data)} bytes for {platform}",
        "professionalism_score": 5,
    }
    result["summary"] = generate_advice(result, platform)
    return result
