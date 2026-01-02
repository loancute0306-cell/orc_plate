import cv2
import os
import re
import time
import numpy as np
from typing import List, Tuple
from paddleocr import PaddleOCR

from .models import PlateDetection, PlateType
from .utils import resize_image, validate_plate_format, format_plate_number


def bbox_to_rect(bbox):
    """
    Convert PaddleOCR bbox (4 points) to rectangle [x1, y1, x2, y2]
    """
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    return [min(xs), min(ys), max(xs), max(ys)]


class TaiwanPlateOCR:
    def __init__(self, disable_model_check: bool = True):
        if disable_model_check:
            os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

        print("🚀 Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=False,
            enable_mkldnn=True,
        )
        print("✅ PaddleOCR initialized")

    def clean_plate_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.upper().replace(" ", "")

        replacements = {
            "O": "0",
            "I": "1",
            "Z": "2",
            "S": "5",
            "B": "8",
            "Q": "0",
            "D": "0",
            "T": "1",
            "G": "6",
            "L": "1",
            "U": "0",
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return re.sub(r"[^A-Z0-9-]", "", text)

    def process_image(
        self,
        image_path: str,
        max_width: int = 500,
        conf_threshold: float = 0.1,
        clean_text: bool = True,
        check_plate_format: bool = True,
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

            print(f"🔍 Processing OCR on image: {image_path}")
            result = self.ocr.ocr(img, cls=False)

            if not result or not result[0]:
                print("❌ No OCR results")
                return plates_detected, image_info

            ocr_lines = result[0]
            print(f"📝 Found {len(ocr_lines)} text regions")

            for bbox, (text, score) in ocr_lines:
                if score < conf_threshold:
                    continue

                raw_text = text
                cleaned = self.clean_plate_text(text) if clean_text else text.upper()
                formatted_plate = format_plate_number(cleaned)

                if check_plate_format:
                    is_valid, plate_type = validate_plate_format(formatted_plate)
                    if not is_valid:
                        continue
                else:
                    plate_type = "unknown"

                rect = bbox_to_rect(bbox)

                plates_detected.append(
                    PlateDetection(
                        plate_number=formatted_plate,
                        confidence=float(score),
                        plate_type=PlateType(plate_type),
                        raw_text=raw_text,
                        position=rect,
                    )
                )

                print(f"✅ Detected plate: {formatted_plate} ({score:.3f})")

            plates_detected.sort(key=lambda x: x.confidence, reverse=True)
            print(f"⏱️ Done in {time.time() - start_time:.2f}s")

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
            result = self.ocr.ocr(img, cls=False)

            if not result or not result[0]:
                return []

            plates_detected: List[PlateDetection] = []

            for bbox, (text, score) in result[0]:
                if score < conf_threshold:
                    continue

                cleaned = self.clean_plate_text(text)
                is_valid, plate_type = validate_plate_format(cleaned)
                if not is_valid:
                    continue

                plates_detected.append(
                    PlateDetection(
                        plate_number=format_plate_number(cleaned),
                        confidence=float(score),
                        plate_type=PlateType(plate_type),
                        raw_text=text,
                        position=bbox_to_rect(bbox),
                    )
                )

            return plates_detected

        except Exception as e:
            print(f"❌ Error processing image bytes: {e}")
            return []
