"""API dua model YOLO untuk greenhouse cabai, diadaptasi dari DuaModel.ipynb."""
import os
import requests
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


HOSTINGER_URL = os.getenv(
    "HOSTINGER_URL",
    "https://smartgreenhouse.pbltifnganjuk.com/api/simpan_deteksi.php"
)

HOSTINGER_API_TOKEN = os.getenv("HOSTINGER_API_TOKEN", "")

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


def gambar_segmentasi(crop_bgr, leaf_result):
    # Salin gambar tanaman
    output = crop_bgr.copy()
    overlay = crop_bgr.copy()

    # Ambil bentuk daun dari hasil YOLO
    polygons = (
        [] if leaf_result.masks is None
        else leaf_result.masks.xy
    )

    rng = np.random.default_rng(123)

    # LANGKAH 1: Berikan warna pada daun
    for polygon in polygons:
        if len(polygon) < 3:
            continue

        pts = np.round(polygon).astype(np.int32)

        color = tuple(
            int(v) for v in rng.integers(60, 256, size=3)
        )

        cv2.fillPoly(overlay, [pts], color)

    # LANGKAH 2: Gabungkan warna transparan
    blended = cv2.addWeighted(
        output, 0.60,
        overlay, 0.40,
        0
    )

    # LANGKAH 3: Gambar garis putih dan nomor merah
    for idx, polygon in enumerate(polygons, start=1):
        if len(polygon) < 3:
            continue

        pts = np.round(polygon).astype(np.int32)

        # Garis putih
        cv2.polylines(
            blended,
            [pts],
            True,
            (255, 255, 255),
            2
        )

        # Cari titik tengah daun
        cx, cy = np.mean(pts, axis=0).astype(int)

        # Nomor merah
        cv2.putText(
            blended,
            str(idx),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return blended

@app.get("/")
def home():
    return {
        "service": "greenhouse-cabai-pcv",
        "status": "online",
        "endpoint": "POST /predict",
        "versi": "APP_PY_TERBARU"
    }

def simpan_ke_hostinger(
    foto,
    foto_segmentasi,
    plants,
    jenis_foto="otomatis",
    id_perintah=0
):
    import json

    if not HOSTINGER_API_TOKEN:
        raise RuntimeError("Token Hostinger belum diatur")

    if not plants:
        raise ValueError("Tidak ada tanaman yang terdeteksi")

    if not foto_segmentasi:
        raise ValueError("Foto segmentasi kosong")

    print(
        "DEBUG KIRIM:",
        "tanaman =", len(plants),
        "foto_segmentasi =", len(foto_segmentasi),
        flush=True
    )

    response = requests.post(
        HOSTINGER_URL,
        headers={
            "X-API-Key": HOSTINGER_API_TOKEN
        },
        files={
            "foto": (
                "cabai_asli.jpg",
                foto,
                "image/jpeg"
            ),
            "foto_segmentasi": (
                "cabai_segmentasi.jpg",
                foto_segmentasi,
                "image/jpeg"
            )
        },
        data={
            "jumlah_daun": str(plants[0]["jumlah_daun"]),
            "status": plants[0]["status"],
            "tanaman": json.dumps(plants),
        
            # Informasi foto dari ESP32-CAM
            "jenis_foto": jenis_foto,
            "id_perintah": str(id_perintah)
        },
        timeout=60
    )

    print("HOSTINGER STATUS:", response.status_code, flush=True)
    print("HOSTINGER RESPONSE:", response.text[:1000], flush=True)

    response.raise_for_status()

    hasil = response.json()

    if not hasil.get("success"):
        raise RuntimeError("Hostinger gagal menyimpan data")

    if not hasil.get("foto_segmentasi_tersimpan"):
        raise RuntimeError("Foto segmentasi belum tersimpan")

    if hasil.get("jumlah_tanaman") != len(plants):
        raise RuntimeError("Jumlah tanaman tersimpan tidak sesuai")

    return hasil

@app.post("/predict")
async def predict(request: Request):
    # Membaca informasi foto dari ESP32-CAM
    jenis_foto = request.headers.get(
        "X-Jenis-Foto", ""
    ).strip().lower()

    id_perintah_raw = request.headers.get(
        "X-ID-Perintah", "0"
    ).strip()

    # Validasi jenis foto
    if jenis_foto not in ("otomatis", "manual"):
        raise HTTPException(
            status_code=400,
            detail="Jenis foto harus otomatis atau manual"
        )

    # Validasi ID perintah
    try:
        id_perintah = int(id_perintah_raw)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="ID perintah harus berupa angka"
        )

    if jenis_foto == "manual" and id_perintah <= 0:
        raise HTTPException(
            status_code=400,
            detail="Foto manual wajib memiliki ID perintah"
        )

    if jenis_foto == "otomatis" and id_perintah != 0:
        raise HTTPException(
            status_code=400,
            detail="Foto otomatis harus menggunakan ID 0"
        )

    print(
        f"[FOTO] Jenis: {jenis_foto}, ID: {id_perintah}",
        flush=True
    )
    
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
        
        # Satu gambar untuk seluruh tanaman
        foto_segmentasi_gabungan = image.copy()
        
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
                                
                # Buat segmentasi untuk tanaman ini
                foto_segmentasi = gambar_segmentasi(
                    crop,
                    leaf_result
                )
                
                # Tentukan posisi crop pada foto asli
                crop_x1 = max(0, x1 - mx)
                crop_y1 = max(0, y1 - my)
                crop_x2 = min(width, x2 + mx)
                crop_y2 = min(height, y2 + my)
                
                # Tempel hasil segmentasi ke foto gabungan
                foto_segmentasi_gabungan[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2
                ] = foto_segmentasi
                
                plants.append({
                    "tanaman_ke": idx,
                    "confidence_tanaman": round(float(box.conf[0].cpu()), 4),
                    "jumlah_daun": count,
                    "status": readiness_status(count),
                    "box_xyxy": [x1, y1, x2, y2]
                })
        
        # Ubah foto segmentasi gabungan menjadi JPEG
        berhasil_segmentasi, jpeg_segmentasi = cv2.imencode(
            ".jpg",
            foto_segmentasi_gabungan,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not berhasil_segmentasi:
            raise RuntimeError("Gagal membuat JPEG segmentasi")

        foto_segmentasi_bytes = jpeg_segmentasi.tobytes()

        # Status awal penyimpanan
        penyimpanan = {
            "success": False,
            "message": "Tanaman tidak terdeteksi"
        }

        if len(plants) > 0:
            tanaman = plants[0]

            # Gunakan JPEG asli jika input berupa JPEG
            content_type = request.headers.get(
                "content-type", ""
            ).split(";")[0].strip().lower()

            if content_type == "image/jpeg":
                foto_jpeg = data
            else:
                berhasil, jpeg = cv2.imencode(
                    ".jpg",
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                if not berhasil:
                    raise RuntimeError("Gagal membuat JPEG")

                foto_jpeg = jpeg.tobytes()


            penyimpanan = simpan_ke_hostinger(
                foto_jpeg,
                foto_segmentasi_bytes,
                plants,
                jenis_foto,
                id_perintah
            )

        return {
            "success": penyimpanan.get("success") is True,
            "saved_to_hostinger": penyimpanan.get("success") is True,
            "jumlah_tanaman_terdeteksi": len(plants),
            "tanaman": plants,
            "penyimpanan": penyimpanan
        }    
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="YOLO berhasil diproses, tetapi komunikasi dengan Hostinger gagal."
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Pemrosesan atau penyimpanan gagal. Periksa log Railway."
        ) from exc

