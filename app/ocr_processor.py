import cv2
import os
import re
import time
from typing import List, Tuple, Optional
from paddleocr import PaddleOCR
from .models import PlateDetection, OCRResult, PlateType
from .utils import resize_image, validate_plate_format, format_plate_number

class TaiwanPlateOCR:
    def __init__(self, disable_model_check: bool = True):
        """Khởi tạo OCR processor"""
        if disable_model_check:
            os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        
        print("🚀 Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            lang='en',
            use_angle_cls=False,
            enable_mkldnn=True,  # Tăng tốc độ trên CPU Intel
        )
        print("✅ PaddleOCR initialized successfully")
    
    def clean_plate_text(self, text: str) -> str:
        """Làm sạch text biển số"""
        if not text:
            return ""
        
        text = str(text).upper().replace(" ", "")
        
        # Sửa các ký tự dễ nhầm lẫn
        replacements = {
            'O': '0', 'I': '1', 'Z': '2', 
            'S': '5', 'B': '8', 'Q': '0',
            'D': '0', 'T': '1', 'G': '6',
            'L': '1', 'U': '0'
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        # Chỉ giữ lại chữ cái, số và dấu gạch ngang
        text = re.sub(r'[^A-Z0-9-]', '', text)
        
        return text
    
    def process_image(
        self, 
        image_path: str,
        max_width: int = 500,
        conf_threshold: float = 0.1,
        clean_text: bool = True,
        check_plate_format: bool = True
    ) -> Tuple[List[PlateDetection], dict]:
        """Xử lý OCR trên ảnh"""
        start_time = time.time()
        plates_detected = []
        image_info = {}
        
        try:
            # Đọc ảnh
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image from {image_path}")
            
            # Lấy thông tin ảnh
            h, w = img.shape[:2]
            image_info = {
                "original_size": f"{w}x{h}",
                "channels": img.shape[2] if len(img.shape) > 2 else 1
            }
            
            # Resize ảnh để xử lý nhanh
            img = resize_image(img, max_width)
            h, w = img.shape[:2]
            image_info["processed_size"] = f"{w}x{h}"
            
            # Thực hiện OCR
            print(f"🔍 Processing OCR on image: {image_path}")
            result = self.ocr.ocr(img, cls=True)
            
            if not result:
                print("❌ No OCR results")
                return plates_detected, image_info
            
            # Xử lý kết quả
            res = result[0]
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            bboxes = res.get('det_boxes', [])
            
            print(f"📝 Found {len(texts)} text regions")
            
            # Xử lý từng kết quả
            for idx, (text, score) in enumerate(zip(texts, scores)):
                if score < conf_threshold:
                    continue
                
                # Làm sạch text
                if clean_text:
                    cleaned = self.clean_plate_text(text)
                else:
                    cleaned = str(text).upper()
                
                # Format biển số
                formatted_plate = format_plate_number(cleaned)
                
                # Kiểm tra định dạng biển số
                if check_plate_format:
                    is_valid, plate_type = validate_plate_format(formatted_plate)
                    if not is_valid:
                        continue
                else:
                    plate_type = "unknown"
                
                # Lấy bounding box nếu có
                bbox = None
                if idx < len(bboxes):
                    bbox = bboxes[idx]
                
                # Tạo PlateDetection object
                plate_detection = PlateDetection(
                    plate_number=formatted_plate,
                    confidence=float(score),
                    plate_type=PlateType(plate_type),
                    raw_text=text,
                    position=bbox
                )
                
                plates_detected.append(plate_detection)
                print(f"✅ Detected plate: {formatted_plate} (confidence: {score:.3f}, type: {plate_type})")
            
            # Sắp xếp theo confidence
            plates_detected.sort(key=lambda x: x.confidence, reverse=True)
            
            processing_time = time.time() - start_time
            print(f"⏱️ Processing completed in {processing_time:.2f} seconds")
            
            return plates_detected, image_info
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return plates_detected, image_info
    
    def process_image_object(
        self, 
        image_bytes: bytes,
        max_width: int = 500,
        conf_threshold: float = 0.1
    ) -> List[PlateDetection]:
        """Xử lý OCR từ bytes ảnh"""
        try:
            # Chuyển bytes thành numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return []
            
            # Resize ảnh
            img = resize_image(img, max_width)
            
            # Thực hiện OCR
            result = self.ocr.ocr(img, cls=True)
            
            if not result:
                return []
            
            plates_detected = []
            res = result[0]
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            
            for text, score in zip(texts, scores):
                if score < conf_threshold:
                    continue
                
                cleaned = self.clean_plate_text(text)
                is_valid, plate_type = validate_plate_format(cleaned)
                
                if is_valid:
                    formatted_plate = format_plate_number(cleaned)
                    plate_detection = PlateDetection(
                        plate_number=formatted_plate,
                        confidence=float(score),
                        plate_type=PlateType(plate_type),
                        raw_text=text
                    )
                    plates_detected.append(plate_detection)
            
            return plates_detected
            
        except Exception as e:
            print(f"❌ Error processing image bytes: {e}")
            return []