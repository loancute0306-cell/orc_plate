import os
import re
import time
import cv2
import numpy as np
from typing import List, Tuple
from paddleocr import PaddleOCR

from .models import PlateDetection, PlateType
from .utils import resize_image, format_plate_number


def bbox_to_rect(bbox):
    """
    Chuyển bbox 4 điểm sang rectangle [x1, y1, x2, y2]
    """
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    return [min(xs), min(ys), max(xs), max(ys)]


class TaiwanPlateOCR:
    def __init__(self, disable_model_check: bool = True, use_local_model: bool = False, model_dir: str = None):
        if disable_model_check:
            os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

        print("🚀 Initializing PaddleOCR...")
        ocr_args = {
            "lang": "en",
            "use_angle_cls": False,
            "enable_mkldnn": True,
        }

        # Nếu dùng model local
        if use_local_model and model_dir:
            ocr_args["det_model_dir"] = os.path.join(model_dir, "det")
            ocr_args["rec_model_dir"] = os.path.join(model_dir, "rec")

        self.ocr = PaddleOCR(**ocr_args)
        print("✅ PaddleOCR initialized")

    @staticmethod
    def clean_plate_text(text: str) -> str:
        """Chuẩn hóa text biển số"""
        if not text:
            return ""
        # Uppercase + remove ký tự lạ
        text = text.upper()
        text = re.sub(r"[^A-Z0-9-]", "", text)
        return text

    def process_image(
        self,
        image_path: str,
        max_width: int = 500,
        conf_threshold: float = 0.1,
    ) -> Tuple[List[PlateDetection], dict]:

        start_time = time.time()
        plates_detected: List[PlateDetection] = []
        image_info = {}

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")

            h, w = img.shape[:2]
            image_info["original_size"] = f"{w}x{h}"

            img = resize_image(img, max_width)
            h, w = img.shape[:2]
            image_info["processed_size"] = f"{w}x{h}"

            # OCR
            result = self.ocr.ocr(img)

            print(f"📝 OCR raw output for {image_path}:")
            for line in result:
                print(line)  # Debug text

            for line in result:
                # Kiểm tra key 'rec_texts'
                rec_texts = line.get('rec_texts', [])
                rec_scores = line.get('rec_scores', [])
                rec_polys = line.get('rec_polys', [])

                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                    score = float(score)
                    if score < conf_threshold:
                        continue

                    cleaned = self.clean_plate_text(text)

                    plates_detected.append(
                        PlateDetection(
                            plate_number=format_plate_number(cleaned),
                            confidence=score,
                            plate_type=PlateType("unknown"),
                            raw_text=text,
                            position=bbox_to_rect(poly),
                        )
                    )

            plates_detected.sort(key=lambda x: x.confidence, reverse=True)
            image_info["plates_count"] = len(plates_detected)
            image_info["processing_time"] = round(time.time() - start_time, 2)

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
        conf_threshold: float = 0.1,
    ) -> List[PlateDetection]:

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return []

            img = resize_image(img, max_width)
            result = self.ocr.ocr(img)

            print("📝 OCR raw output for image bytes:")
            for line in result:
                print(line)  # Debug

            plates_detected: List[PlateDetection] = []

            for line in result:
                if isinstance(line, dict):
                    bbox = line.get("position")
                    text = line.get("text", "")
                    score = float(line.get("confidence", 0))
                else:
                    bbox = line[0]
                    text, score = line[1]
                    score = float(score)

                if score < conf_threshold or not text or not bbox:
                    continue

                cleaned = self.clean_plate_text(text)

                plates_detected.append(
                    PlateDetection(
                        plate_number=format_plate_number(cleaned),
                        confidence=score,
                        plate_type=PlateType("unknown"),
                        raw_text=text,
                        position=bbox_to_rect(bbox),
                    )
                )

            return plates_detected

        except Exception as e:
            print(f"❌ Error processing image bytes: {e}")
            import traceback
            traceback.print_exc()
            return []
