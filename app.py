"""API dua model YOLO untuk greenhouse cabai, diadaptasi dari DuaModel.ipynb."""
import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from ultralytics import YOLO

app = FastAPI(title="Greenhouse Cabai - YOLO API")

PLANT_CONF = 0.25
LEAF_CONF = 0.25
IMAGE_SIZE = 640
CROP_MARGIN = 0.10
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def readiness_status(n_leaves: int) -> str:
    # Kriteria operasional awal dari notebook; bukan diagnosis agronomis.
    if 4 <= n_leaves <= 6:
        return "Siap (4-6 daun)"
    if n_leaves < 4:
        return "Belum siap (<4 daun)"
    return "Perlu pemeriksaan (>6 daun)"


@lru_cache(maxsize=1)
def load_models():
    plant_path = os.getenv("PLANT_MODEL", "best_plant.pt")
    leaf_path = os.getenv("LEAF_MODEL", "best_cabai.pt")
    plant_model = YOLO(plant_path)
    leaf_model = YOLO(leaf_path)
    if plant_model.task != "detect" or leaf_model.task != "segment":
        raise RuntimeError("Model harus berurutan: detection tanaman dan segmentation daun")
    return plant_model, leaf_model


@app.get("/")
def home():
    return {"service": "greenhouse-cabai-pcv", "status": "online", "endpoint": "POST /predict"}


@app.post("/predict")
async def predict(request: Request):
    # Kirim body sebagai JPEG mentah (Content-Type: image/jpeg).
    if request.headers.get("content-type", "").split(";")[0].strip() not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Gunakan Content-Type: image/jpeg atau image/png")
    data = await request.body()
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(400, "Gambar kosong atau melebihi 8 MB")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Gambar tidak dapat dibaca")

    try:
        plant_model, leaf_model = load_models()
        height, width = image.shape[:2]
        detection = plant_model.predict(source=image, conf=PLANT_CONF, imgsz=IMAGE_SIZE,
                                        device="cpu", verbose=False)[0]
        plants = []
        if detection.boxes is not None:
            for idx, box in enumerate(detection.boxes, start=1):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, x2 = max(0, min(int(x1), width - 1)), max(0, min(int(x2), width))
                y1, y2 = max(0, min(int(y1), height - 1)), max(0, min(int(y2), height))
                if x2 <= x1 or y2 <= y1:
                    continue
                mx, my = int((x2 - x1) * CROP_MARGIN), int((y2 - y1) * CROP_MARGIN)
                crop = image[max(0, y1-my):min(height, y2+my),
                             max(0, x1-mx):min(width, x2+mx)].copy()
                if crop.size == 0:
                    continue
                leaf_result = leaf_model.predict(source=crop, conf=LEAF_CONF,
                                                 imgsz=IMAGE_SIZE, retina_masks=True,
                                                 device="cpu", verbose=False)[0]
                polygons = [] if leaf_result.masks is None else leaf_result.masks.xy
                count = len(polygons)
                plants.append({
                    "tanaman_ke": idx,
                    "confidence_tanaman": round(float(box.conf[0].cpu()), 4),
                    "jumlah_daun": count,
                    "status": readiness_status(count),
                    "box_xyxy": [x1, y1, x2, y2]
                })
        return {"jumlah_tanaman_terdeteksi": len(plants), "tanaman": plants}
    except Exception as exc:
        # Hindari mengembalikan detail internal server ke pengguna API.
        raise HTTPException(500, "Prediksi gagal. Periksa log server dan ketersediaan RAM.") from exc
