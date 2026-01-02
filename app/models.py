from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class PlateType(str, Enum):
    REGULAR = "regular"
    COMMERCIAL = "commercial"
    TAXI = "taxi"
    TEMPORARY = "temporary"
    GOVERNMENT = "government"
    UNKNOWN = "unknown"

class OCRResult(BaseModel):
    """Kết quả OCR cho một dòng text"""
    text: str
    confidence: float
    bbox: Optional[List[List[int]]] = None  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    cleaned_text: str

class PlateDetection(BaseModel):
    """Thông tin biển số phát hiện"""
    plate_number: str
    confidence: float
    plate_type: PlateType
    raw_text: str
    position: Optional[List[int]] = None  # [x, y, w, h]
    timestamp: datetime = Field(default_factory=datetime.now)

class OCRRequest(BaseModel):
    """Request model cho API"""
    image_path: Optional[str] = None  # Đường dẫn ảnh local
    image_url: Optional[str] = None   # URL ảnh
    max_width: int = 500
    conf_threshold: float = 0.1
    clean_text: bool = True
    check_plate_format: bool = True

class OCRResponse(BaseModel):
    """Response model cho API"""
    success: bool
    message: str
    plates_detected: List[PlateDetection] = []
    total_plates: int = 0
    processing_time: float
    image_info: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)