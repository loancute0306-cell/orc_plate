from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import time
import shutil
import cv2
import numpy as np
from typing import Optional
from datetime import datetime
import requests

from app.models import OCRResponse, PlateDetection
from app.ocr_processor import TaiwanPlateOCR
from app.utils import resize_image, format_plate_number

# ================== FastAPI App ==================
app = FastAPI(
    title="Taiwan License Plate OCR API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục upload tạm
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Khởi tạo OCR processor
ocr_processor = TaiwanPlateOCR(disable_model_check=True)

# ================== ROUTES ==================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🚗 Taiwan License Plate OCR API",
        "version": "1.0.0",
        "endpoints": {
            "upload_image": "/api/ocr/upload",
            "process_url": "/api/ocr/process-url",
            "health": "/api/health"
        },
        "docs": "/docs"
    }

@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "taiwan-plate-ocr"
    }

# ---------------- Upload & OCR ----------------
@app.post("/api/ocr/upload", tags=["OCR"])
async def upload_and_ocr(
    file: UploadFile = File(..., description="Image file to process"),
    max_width: int = Query(500, ge=100, le=2000),
    conf_threshold: float = Query(0.1, ge=0.0, le=1.0)
):
    start_time = time.time()

    # Kiểm tra định dạng
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    try:
        # Lưu file tạm
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        plates_detected, image_info = ocr_processor.process_image(
            temp_path,
            max_width=max_width,
            conf_threshold=conf_threshold
        )

        # Xóa file tạm
        os.remove(temp_path)

        processing_time = time.time() - start_time

        return OCRResponse(
            success=True,
            message="OCR completed successfully",
            plates_detected=plates_detected,
            total_plates=len(plates_detected),
            processing_time=round(processing_time, 2),
            image_info={**image_info, "original_filename": file.filename}
        )

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- OCR từ URL ----------------
@app.post("/api/ocr/process-url", tags=["OCR"])
async def ocr_from_url(
    image_url: str = Query(..., description="URL of image"),
    max_width: int = Query(500, ge=100, le=2000),
    conf_threshold: float = Query(0.1, ge=0.0, le=1.0)
):
    start_time = time.time()
    try:
        # Tải ảnh từ URL
        resp = requests.get(image_url, timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Cannot download image from URL")

        image_bytes = resp.content
        plates_detected = ocr_processor.process_image_object(
            image_bytes=image_bytes,
            max_width=max_width,
            conf_threshold=conf_threshold
        )

        processing_time = time.time() - start_time

        return OCRResponse(
            success=True,
            # message="OCR from URL completed successfully",
            plates_detected=plates_detected,
            # total_plates=len(plates_detected),
            processing_time=round(processing_time, 2),
            # image_info={"source_url": image_url}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
