"""FastAPI inference for the trained ResNet-18 COVID CXR classifier."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES
from src.gradcam import GradCAM, image_to_b64_png, overlay_cam_on_image, zone_scores
from src.model import get_resnet18

app = FastAPI(
    title="PulmoScan AI",
    description="4-class chest X-ray classifier (ResNet-18) with Grad-CAM lung-zone visualization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS = ROOT / "artifacts"
if ARTIFACTS.exists():
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS)), name="artifacts")

SAMPLES = ROOT / "sample_images"
if SAMPLES.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES)), name="samples")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_CANDIDATES = [
    ROOT / "backend" / "resnet18_covid_best.pth",
    ROOT / "models" / "resnet18_covid.pth",
]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_model():
    model = get_resnet18(num_classes=len(CLASSES), pretrained=False)
    weights_path = next((p for p in WEIGHT_CANDIDATES if p.exists()), None)
    loaded = False
    if weights_path is not None:
        try:
            state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        loaded = True
    model.to(DEVICE)
    model.eval()
    return model, weights_path, loaded


MODEL, WEIGHTS_PATH, MODEL_LOADED = _load_model()

METRICS_PATH = ARTIFACTS / "metrics.json"
METRICS = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = ROOT / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>PulmoScan AI</h1><p>Frontend not found.</p>"

@app.get("/training.html", response_class=HTMLResponse)
def training_page():
    html_path = ROOT / "frontend" / "training.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Training Results</h1><p>Page not found.</p>"

@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "name": "PulmoScan AI",
        "model_loaded": MODEL_LOADED,
        "weights": str(WEIGHTS_PATH) if WEIGHTS_PATH else None,
        "device": str(DEVICE),
        "classes": CLASSES,
    }

@app.get("/health")
def health():
    return status()


@app.get("/metrics")
def metrics():
    return METRICS


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Trained weights are not loaded.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)

    pred_i = int(pred_idx.item())
    predicted_class = CLASSES[pred_i]
    prob_dict = {CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASSES))}

    cam = GradCAM(MODEL, MODEL.layer4)
    try:
        heatmap = cam(tensor, class_idx=pred_i)
    finally:
        cam.remove_hooks()

    overlay = overlay_cam_on_image(image, heatmap)
    heat_img = Image.fromarray((heatmap * 255).astype("uint8")).resize(image.size)
    zones = zone_scores(heatmap)

    # For Normal, keep 3D highlight mild — CAM still shown on the 2D overlay.
    disease_mass = 1.0 - float(probs[CLASSES.index("Normal")])
    scaled_zones = {k: round(v * disease_mass, 4) for k, v in zones.items()}

    return {
        "predicted_class": predicted_class,
        "confidence": round(float(confidence) * 100, 2),
        "probabilities": prob_dict,
        "zones": scaled_zones,
        "zones_raw": zones,
        "heatmap_png": image_to_b64_png(heat_img.convert("RGB")),
        "overlay_png": image_to_b64_png(overlay),
        "disclaimer": (
            "Research screening aid only. Grad-CAM zones are 2D attention mapped onto "
            "a 3D lung illustration, not CT segmentation or a clinical diagnosis."
        ),
        "model": {
            "architecture": "ResNet-18",
            "loaded": MODEL_LOADED,
            "test_accuracy": METRICS.get("test_accuracy"),
            "test_macro_f1": METRICS.get("test_macro_f1"),
        },
    }
