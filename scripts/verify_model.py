"""Download and verify the pretrained fruit classifier from Hugging Face."""
import json
import time

import torch
import torchvision.models as models
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

REPO_ID = "bhumong/fruit-classifier-efficientnet-b0"
MODEL_FILE = "pytorch_model.bin"
CONFIG_FILE = "config.json"


def load_model(repo_id=REPO_ID):
    config_path = hf_hub_download(repo_id=repo_id, filename=CONFIG_FILE)
    with open(config_path) as f:
        config = json.load(f)

    num_labels = config["num_labels"]
    id2label = {int(k): v for k, v in config["id2label"].items()}

    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_labels)

    model_path = hf_hub_download(repo_id=repo_id, filename=MODEL_FILE)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, id2label


def preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(model, id2label, image: Image.Image, top_k=3):
    tf = preprocess()
    tensor = tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_idx = torch.topk(probs, top_k)
    return [
        {"class": id2label[int(i)], "confidence": float(p)}
        for p, i in zip(top_probs, top_idx)
    ]


if __name__ == "__main__":
    t0 = time.time()
    model, config, id2label = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"Classes: {len(id2label)}")
