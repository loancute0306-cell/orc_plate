import re
import os
import cv2
import numpy as np
from typing import Tuple, Optional
from urllib.parse import urlparse
import requests
from PIL import Image
import io

def download_image_from_url(url: str, timeout: int = 10) -> Optional[np.ndarray]:
    """Tải ảnh từ URL"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Chuyển bytes thành numpy array
        image = Image.open(io.BytesIO(response.content))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Error downloading image from URL: {e}")
        return None

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load ảnh từ đường dẫn local"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        return img
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def resize_image(img: np.ndarray, max_width: int = 500) -> np.ndarray:
    """Resize ảnh để xử lý nhanh hơn"""
    if img is None:
        return img
    
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_width = max_width
        new_height = int(h * scale)
        img = cv2.resize(img, (new_width, new_height))
        print(f"✅ Image resized: {w}x{h} -> {new_width}x{new_height}")
    
    return img

def get_image_info(img: np.ndarray) -> dict:
    """Lấy thông tin ảnh"""
    if img is None:
        return {}
    
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(img.dtype)
    }

def validate_plate_format(plate_number: str) -> Tuple[bool, str]:
    """Kiểm tra định dạng biển số Taiwan"""
    if not plate_number:
        return False, "Empty"
    
    # Chuẩn hóa: loại bỏ ký tự đặc biệt, chuyển hoa
    clean_plate = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
    
    if len(clean_plate) < 4 or len(clean_plate) > 8:
        return False, "invalid_length"
    
    # Phân tích thành phần
    letters = sum(1 for c in clean_plate if c.isalpha())
    digits = sum(1 for c in clean_plate if c.isdigit())
    
    if letters < 1 or digits < 2:
        return False, "invalid_composition"
    
    # Xác định loại biển số
    plate_type = detect_plate_type(clean_plate)
    
    return True, plate_type

def detect_plate_type(plate_number: str) -> str:
    """Xác định loại biển số"""
    # Các mẫu biển số Taiwan thông dụng
    patterns = {
        "regular": [  # Xe thường: XX-1234, XXX-123
            r'^[A-Z]{2,3}-?\d{3,4}$',
            r'^[A-Z]{2,3}\d{3,4}$'
        ],
        "commercial": [  # Xe thương mại: 123-XX
            r'^\d{2,3}-?[A-Z]{1,3}$',
            r'^\d{2,3}[A-Z]{1,3}$'
        ],
        "taxi": [  # Taxi: XXX-123
            r'^[A-Z]{3}-?\d{3}$',
            r'^[A-Z]{3}\d{3}$'
        ],
        "temporary": [  # Biển tạm: 123-456
            r'^\d{3}-?\d{3}$',
            r'^\d{6}$'
        ],
        "government": [  # Xe chính phủ: XXX-1234
            r'^[A-Z]{3}-?\d{4}$',
            r'^[A-Z]{3}\d{4}$'
        ]
    }
    
    for plate_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.match(pattern, plate_number, re.IGNORECASE):
                return plate_type
    
    return "unknown"

def format_plate_number(plate_text: str) -> str:
    """Format biển số cho đẹp"""
    if not plate_text:
        return ""
    
    # Loại bỏ ký tự không phải chữ số
    clean = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Thêm dấu gạch ngang nếu cần
    if len(clean) == 6:
        # XXX-123 hoặc 123-XXX
        if clean[:3].isalpha() and clean[3:].isdigit():
            return f"{clean[:3]}-{clean[3:]}"
        elif clean[:3].isdigit() and clean[3:].isalpha():
            return f"{clean[:3]}-{clean[3:]}"
    elif len(clean) == 7:
        # XX-1234
        if clean[:2].isalpha() and clean[2:].isdigit():
            return f"{clean[:2]}-{clean[2:]}"
    
    return clean