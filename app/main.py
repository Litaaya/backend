from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import torch
import timm
import torch.nn as nn
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- App setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
IMG_SIZE = 224
NUM_CLASSES = 7
DEVICE = 'cpu'
CLASSES = ['Casual', 'MajorBrand', 'NoClothes', 'Publications', 'Runway', 'Traditional', 'Trash']

MODEL_PATH = "app/model/resnet50_final_best.pth"
RUNWAY_EMBED_PATH = "app/embeds/runway_embeds.npy"
PUBLIC_EMBED_PATH = "app/embeds/publications_embeds.npy"

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load model ---
model = timm.create_model('resnet50', pretrained=False, num_classes=NUM_CLASSES)
in_features = model.get_classifier().in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def extract_features(x):
    x = model.forward_features(x)
    return x.mean(dim=[2, 3])  # Global average pooling

# --- Load embedding sets ---
runway_embeds = np.load(RUNWAY_EMBED_PATH)
public_embeds = np.load(PUBLIC_EMBED_PATH)

def analyze_trend(user_emb: np.ndarray):
    sim_runway = cosine_similarity([user_emb], runway_embeds).mean()
    sim_public = cosine_similarity([user_emb], public_embeds).mean()

    avg_sim = max(sim_runway, sim_public)

    if avg_sim < 0.3:
        level = "LOW"
    else:
        level = "MAYBE"

    closer_to = "Runway" if sim_runway >= sim_public else "Publications"

    return {
        "runway_similarity": round(float(sim_runway), 4),
        "publications_similarity": round(float(sim_public), 4),
        "assessment": level,
        "closer_to": closer_to
    }

def read_image(file) -> torch.Tensor:
    image = Image.open(BytesIO(file)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_tensor = read_image(img_bytes).to(DEVICE)

    # --- Predict class ---
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    # --- Extract embedding ---
    with torch.no_grad():
        emb = extract_features(img_tensor).squeeze().cpu().numpy()

    # --- Analyze trend ---
    trend_score = analyze_trend(emb)

    return {
        "predicted_class": CLASSES[pred],
        "probabilities": {CLASSES[i]: round(float(probs[0][i]), 4) for i in range(len(CLASSES))},
        "trend_score": trend_score
    }
