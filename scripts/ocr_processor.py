"""
ocr_processor.py - Extrage text din imagini folosind PaddleOCR

Procesează imagini din PDF-uri și extrage text folosind OCR.
Suportă limba română + engleză.

PaddleOCR:
- Gratuit și open-source
- Funcționează pe CPU și GPU
- Suportă 80+ limbi incluisnd română
- Mai bun decât Tesseract pentru imagini educaționale
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# Importă PaddleOCR
try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError(
        "PaddleOCR not installed. Run: pip install paddleocr"
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Rezultat al OCR pentru o imagine"""
    text: str  # Text extras
    confidence: float  # Confidență generală (0.0 - 1.0)
    lines: List[Dict]  # Detalii pentru fiecare linie detecată
    language: str  # Limba detectată
    processing_time_ms: float  # Timp procesare în ms


class OCRProcessor:
    """
    Procesor OCR cu suport multilingual și optimizări pentru imagini educaționale.

    Utilizare:
        processor = OCRProcessor(languages=['ro', 'en'])
        result = processor.process_image(image_array)
        print(result.text)
    """

    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        """
        Inițializează PaddleOCR.

        Args:
            languages: Lista de limbi ('ro', 'en', etc). Default: ['ro', 'en']
            use_gpu: Să folosești GPU dacă disponibil. Default: False (CPU e mai stabil)
        """
        if languages is None:
            languages = ['ro', 'en']

        self.languages = languages

        logger.info(f"Initializing PaddleOCR with languages: {languages}")
        logger.info(f"GPU enabled: {use_gpu}")

        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Detectează text rotit
                lang=languages,
                use_gpu=use_gpu,
                show_log=False  # Suppress verbozitate
            )
            logger.info("✅ PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def process_image(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5,
        priority_threshold: float = 0.8
    ) -> OCRResult:
        """
        Procesează o imagine și extrage text.

        Args:
            image: Imagine ca numpy array (din PyMuPDF sau PIL)
            min_confidence: Minim confidence pentru a include text (0.0 - 1.0)
            priority_threshold: Doar procesează dacă priority_score > threshold

        Returns:
            OCRResult cu text și metadate

        Exemplu:
            >>> import cv2
            >>> img = cv2.imread("diagram.png")
            >>> result = processor.process_image(img)
            >>> print(result.text)
            "Aria pătrat = latura²"
            >>> print(f"Confidence: {result.confidence:.2%}")
            "Confidence: 92.34%"
        """

        import time
        start_time = time.time()

        try:
            # Rulează OCR
            ocr_result = self.ocr.ocr(image, cls=True)

            # Parsează rezultate
            extracted_lines = []
            confidences = []

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]  # Textul detectat
                    confidence = line[1][1]  # Confidence score

                    # Filtrează text cu confidence joasă
                    if confidence >= min_confidence:
                        extracted_lines.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': line[0]  # Bounding box pentru text
                        })
                        confidences.append(confidence)

            # Calculează confidence medie
            avg_confidence = (
                np.mean(confidences) if confidences else 0.0
            )

            # Combină liniile în text complet
            full_text = "\n".join([line['text'] for line in extracted_lines])

            processing_time = (time.time() - start_time) * 1000  # ms

            logger.debug(
                f"OCR completed: {len(extracted_lines)} lines, "
                f"confidence: {avg_confidence:.2%}, time: {processing_time:.1f}ms"
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                lines=extracted_lines,
                language="mixed" if len(self.languages) > 1 else self.languages[0],
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                lines=[],
                language="unknown",
                processing_time_ms=0.0
            )

    def process_images_selective(
        self,
        images: List[Dict],
        priority_threshold: float = 0.8
    ) -> List[Dict]:
        """
        Procesează mai multe imagini cu selective OCR.

        Doar procesează imagini cu priority_score >= threshold.
        Economisește timp și resursă ignoring imagini decorative.

        Args:
            images: Listă de imagini (din pdf_extractor.py)
            priority_threshold: Minimum priority score pentru OCR

        Returns:
            Lista de imagini cu text extras (dacă > threshold)

        Exemplu:
            >>> from pdf_extractor import extract_text_and_images
            >>> result = extract_text_and_images("manual.pdf")
            >>> processed = processor.process_images_selective(result.images, threshold=0.7)
            >>> print(f"Processed {len(processed)} high-priority images")
        """

        processed_images = []

        for i, img_data in enumerate(images):
            priority = img_data.get('priority_score', 0.5)

            # Sări peste imagini cu prioritate joasă
            if priority < priority_threshold:
                logger.debug(
                    f"Skipping image {i} (priority: {priority:.2f} < {priority_threshold})"
                )
                continue

            try:
                # În context real, imaginea e deja bytes din PDF
                # Aici simulez procesarea
                ocr_result = self.process_image(
                    img_data.get('image_array'),
                    min_confidence=0.5
                )

                processed_images.append({
                    'original_data': img_data,
                    'ocr_text': ocr_result.text,
                    'ocr_confidence': ocr_result.confidence,
                    'processing_time_ms': ocr_result.processing_time_ms
                })

            except Exception as e:
                logger.warning(f"Failed to process image {i}: {e}")
                continue

        logger.info(
            f"Processed {len(processed_images)}/{len(images)} images "
            f"(threshold: {priority_threshold})"
        )

        return processed_images


def batch_process_images(
    images: List[np.ndarray],
    languages: List[str] = None,
    use_gpu: bool = False,
    min_confidence: float = 0.5
) -> List[OCRResult]:
    """
    Procesează batch de imagini cu o singură instanță OCR.

    Eficient pentru procesare multilă imagini (reutilizează model).

    Args:
        images: Lista de imagini numpy arrays
        languages: Limbi pentru OCR
        use_gpu: Să folosești GPU
        min_confidence: Minim confidence filter

    Returns:
        Lista de OCRResult objects

    Exemplu:
        >>> import cv2
        >>> images = [cv2.imread(f"diagram_{i}.png") for i in range(10)]
        >>> results = batch_process_images(images)
        >>> for result in results:
        ...     print(result.text[:50])
    """

    processor = OCRProcessor(languages=languages, use_gpu=use_gpu)

    results = []
    for i, image in enumerate(images):
        logger.info(f"Processing image {i+1}/{len(images)}")
        result = processor.process_image(image, min_confidence=min_confidence)
        results.append(result)

    return results


# Exemplu de utilizare
if __name__ == "__main__":
    import cv2
    import os
    from pathlib import Path

    # Inițializează processor
    processor = OCRProcessor(languages=['ro', 'en'], use_gpu=False)

    # Caută imagini test
    test_images_dir = Path(__file__).parent.parent / "materiale_didactice"

    if test_images_dir.exists():
        # Caută fișiere imagine
        image_files = list(test_images_dir.glob("**/*.png")) + \
                     list(test_images_dir.glob("**/*.jpg"))

        if image_files:
            print(f"Found {len(image_files)} test images")
            # Procesează prima imagine
            test_image = cv2.imread(str(image_files[0]))
            if test_image is not None:
                result = processor.process_image(test_image)
                print(f"OCR Text: {result.text[:100]}")
                print(f"Confidence: {result.confidence:.2%}")
                print(f"Processing time: {result.processing_time_ms:.1f}ms")
        else:
            print("No image files found in test directory")
    else:
        print("Test directory not found")
