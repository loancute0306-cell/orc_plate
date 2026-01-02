from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import time
import shutil
from typing import Optional, List
from datetime import datetime

from .models import OCRRequest, OCRResponse, PlateDetection
from .ocr_processor import TaiwanPlateOCR
from .utils import download_image_from_url, load_image, resize_image

# Khởi tạo FastAPI app
app = FastAPI(
    title="Taiwan License Plate OCR API",
    description="API nhận dạng biển số xe Taiwan sử dụng PaddleOCR",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo OCR processor
ocr_processor = TaiwanPlateOCR(disable_model_check=True)

# Tạo thư mục uploads nếu chưa tồn tại
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== ROUTES ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "🚗 Taiwan License Plate OCR API",
        "version": "1.0.0",
        "endpoints": {
            "upload_image": "/api/ocr/upload",
            "process_path": "/api/ocr/process",
            "process_url": "/api/ocr/process-url",
            "health": "/api/health",
            "test": "/api/test"
        },
        "documentation": "/docs"
    }

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "taiwan-plate-ocr"
    }

@app.get("/api/test", tags=["Test"])
async def test_ocr():
    """Test endpoint với ảnh mẫu"""
    sample_image = "data/images/xem.jpg"
    
    if not os.path.exists(sample_image):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Sample image not found",
                "suggestions": ["Place test images in data/images/ folder"]
            }
        )
    
    start_time = time.time()
    
    try:
        plates_detected, image_info = ocr_processor.process_image(
            image_path=sample_image,
            max_width=500,
            conf_threshold=0.1
        )
        
        processing_time = time.time() - start_time
        
        response = OCRResponse(
            success=True,
            message="OCR test completed successfully",
            plates_detected=plates_detected,
            total_plates=len(plates_detected),
            processing_time=processing_time,
            image_info=image_info
        )
        
        return response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error during OCR test: {str(e)}",
                "processing_time": time.time() - start_time
            }
        )

@app.post("/api/ocr/upload", response_model=OCRResponse, tags=["OCR"])
async def upload_and_process_image(
    file: UploadFile = File(..., description="Image file to process"),
    max_width: int = Query(500, ge=100, le=2000, description="Maximum width for image processing"),
    conf_threshold: float = Query(0.1, ge=0.0, le=1.0, description="Confidence threshold for text detection"),
    clean_text: bool = Query(True, description="Clean and normalize plate text")
):
    """Upload ảnh và xử lý OCR"""
    start_time = time.time()
    
    # Kiểm tra file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Tạo tên file unique
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    try:
        # Lưu file tạm
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📁 File saved: {file_path}")
        
        # Xử lý OCR
        plates_detected, image_info = ocr_processor.process_image(
            image_path=file_path,
            max_width=max_width,
            conf_threshold=conf_threshold,
            clean_text=clean_text
        )
        
        processing_time = time.time() - start_time
        
        # Xóa file tạm
        os.remove(file_path)
        
        # Tạo response
        response = OCRResponse(
            success=True,
            message="OCR processing completed successfully",
            plates_detected=plates_detected,
            total_plates=len(plates_detected),
            processing_time=processing_time,
            image_info={
                **image_info,
                "original_filename": file.filename,
                "file_size": file.size
            }
        )
        
        return response
        
    except Exception as e:
        # Dọn dẹp file tạm nếu có lỗi
        if os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

# @app.post("/api/ocr/process", response_model=OCRResponse, tags=["OCR"])
# async def process_image_from_path(request: OCRRequest):
#     """Xử lý OCR từ đường dẫn ảnh local"""
#     start_time = time.time()
    
#     if not request.image_path:
#         raise HTTPException(
#             status_code=400,
#             detail="Please provide image_path in request body"
#         )
    
#     if not os.path.exists(request.image_path):
#         raise HTTPException(
#             status_code=404,
#             detail=f"Image not found at path: {request.image_path}"
#         )
    
#     try:
#         plates_detected, image_info = ocr_processor.process_image(
#             image_path=request.image_path,
#             max_width=request.max_width,
#             conf_threshold=request.conf_threshold,
#             clean_text=request.clean_text,
#             check_plate_format=request.check_plate_format
#         )
        
#         processing_time = time.time() - start_time
        
#         response = OCRResponse(
#             success=True,
#             message="OCR processing completed successfully",
#             plates_detected=plates_detected,
#             total_plates=len(plates_detected),
#             processing_time=processing_time,
#             image_info=image_info
#         )
        
#         return response
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing image: {str(e)}"
#         )

# @app.post("/api/ocr/process-url", response_model=OCRResponse, tags=["OCR"])
# async def process_image_from_url(
#     image_url: str = Query(..., description="URL of the image to process"),
#     max_width: int = Query(500, ge=100, le=2000),
#     conf_threshold: float = Query(0.1, ge=0.0, le=1.0)
# ):
#     """Xử lý OCR từ URL ảnh"""
#     start_time = time.time()
    
#     try:
#         # Tải ảnh từ URL
#         from .utils import download_image_from_url
#         img = download_image_from_url(image_url)
        
#         if img is None:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Cannot download image from URL"
#             )
        
#         # Lưu ảnh tạm
#         file_id = str(uuid.uuid4())
#         temp_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
#         cv2.imwrite(temp_path, img)
        
#         # Xử lý OCR
#         plates_detected, image_info = ocr_processor.process_image(
#             image_path=temp_path,
#             max_width=max_width,
#             conf_threshold=conf_threshold
#         )
        
#         # Xóa file tạm
#         os.remove(temp_path)
        
#         processing_time = time.time() - start_time
        
#         response = OCRResponse(
#             success=True,
#             message="OCR processing from URL completed successfully",
#             plates_detected=plates_detected,
#             total_plates=len(plates_detected),
#             processing_time=processing_time,
#             image_info={
#                 **image_info,
#                 "source_url": image_url
#             }
#         )
        
#         return response
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing image from URL: {str(e)}"
#         )

@app.get("/api/sample-images", tags=["Samples"])
async def get_sample_images():
    """Lấy danh sách ảnh mẫu có sẵn"""
    sample_dir = "data/images"
    
    if not os.path.exists(sample_dir):
        return {
            "available": False,
            "message": "Sample images directory not found"
        }
    
    images = []
    for filename in os.listdir(sample_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filepath = os.path.join(sample_dir, filename)
            file_size = os.path.getsize(filepath)
            images.append({
                "filename": filename,
                "path": filepath,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            })
    
    return {
        "available": True,
        "count": len(images),
        "directory": sample_dir,
        "images": images
    }

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Khởi tạo khi app bắt đầu"""
    print("🚀 Starting Taiwan License Plate OCR API...")
    print(f"📁 Upload directory: {os.path.abspath(UPLOAD_DIR)}")
    
    # Kiểm tra thư mục uploads
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    
    print("✅ API is ready!")

# Chạy server nếu chạy trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )