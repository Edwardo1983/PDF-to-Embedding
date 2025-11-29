"""
pdf_extractor.py - Extrage text și imagini din PDF-uri folosind PyMuPDF

Această funcție:
1. Extrage text din fiecare pagină
2. Identifică și salvează imagini (>50KB)
3. Detectează diagrame educaționale
4. Colectează metadate (pagini, mărime, etc)

PyMuPDF e 10x mai rapid decât PyPDF2 pentru PDF-uri mari.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF
import hashlib
from dataclasses import dataclass

# Configure logging în limba română
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """Clasă pentru a stoca informații despre o imagine din PDF"""
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) - coordonate
    size_bytes: int  # Mărimea imaginii în bytes
    priority_score: float  # 0.0 - 1.0 pentru OCR
    image_type: str  # "diagram", "photo", "chart", etc


@dataclass
class PDFExtractionResult:
    """Rezultat complet al extracției dintr-un PDF"""
    pdf_path: str
    text: str  # Text complet extras
    images: List[ImageData]  # Lista imagini detectate
    total_pages: int
    file_size_bytes: int
    extraction_status: str  # "success" sau "error"
    error_message: Optional[str] = None


def extract_text_and_images(pdf_path: str, max_pages: Optional[int] = None) -> PDFExtractionResult:
    """
    Extrage text și imagini dintr-un PDF folosind PyMuPDF.

    Args:
        pdf_path: Path la fișierul PDF

    Returns:
        PDFExtractionResult cu text, imagini și metadate

    Exemplu:
        >>> result = extract_text_and_images("manuals/clasa_1/matematica.pdf")
        >>> print(f"Text extras: {len(result.text)} caractere")
        >>> print(f"Imagini găsite: {len(result.images)}")
    """

    try:
        pdf_path = str(pdf_path)  # Convert Path object to string if needed

        # Verifică dacă fișierul există
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return PDFExtractionResult(
                pdf_path=pdf_path,
                text="",
                images=[],
                total_pages=0,
                file_size_bytes=0,
                extraction_status="error",
                error_message=f"File not found: {pdf_path}"
            )

        # Deschide PDF-ul cu PyMuPDF
        pdf_document = fitz.open(pdf_path)

        # Colectează informații generale
        file_size_bytes = Path(pdf_path).stat().st_size
        total_pages = len(pdf_document)
        pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

        logger.info(f"Proceeding PDF: {Path(pdf_path).name} ({pages_to_process}/{total_pages} pages, {file_size_bytes / 1024 / 1024:.1f}MB)")

        # Extrage text din toate paginile
        all_text = []
        images_found = []

        for page_number in range(pages_to_process):
            try:
                page = pdf_document[page_number]

                # Extrage text din pagină
                page_text = page.get_text()
                all_text.append(page_text)

                # Detectează imagini în pagina
                image_list = page.get_images()

                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]  # Image reference ID

                        # Extrage imaginea
                        pix = fitz.Pixmap(pdf_document, xref)

                        # Calculează mărimea în bytes
                        image_bytes = pix.tobytes()
                        image_size = len(image_bytes)

                        # Filtrează imagini prea mici (nu sunt relevante)
                        min_image_size = 51200  # 50KB
                        if image_size < min_image_size:
                            continue

                        # Obține bounding box pentru imagine
                        image_rect = page.get_image_rects(img_info)
                        if image_rect:
                            bbox = image_rect[0]
                        else:
                            bbox = (0, 0, 100, 100)  # Default

                        # Calculează priority score (pentru OCR)
                        priority = _calculate_image_priority(pix, image_size)

                        # Creează ImageData object
                        img_data = ImageData(
                            page_number=page_number + 1,
                            bbox=bbox,
                            size_bytes=image_size,
                            priority_score=priority,
                            image_type=_detect_image_type(pix)
                        )

                        images_found.append(img_data)

                    except Exception as e:
                        logger.warning(f"Could not extract image {img_index} from page {page_number + 1}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error extracting text from page {page_number + 1}: {e}")
                continue

        # Combină textul din toate paginile
        combined_text = "\n".join(all_text)

        # Închide PDF-ul
        pdf_document.close()

        logger.info(f"✅ Successfully extracted: {len(combined_text)} chars, {len(images_found)} images")

        return PDFExtractionResult(
            pdf_path=pdf_path,
            text=combined_text,
            images=images_found,
            total_pages=pages_to_process,
            file_size_bytes=file_size_bytes,
            extraction_status="success"
        )

    except Exception as e:
        logger.error(f"Unexpected error processing {pdf_path}: {e}")
        return PDFExtractionResult(
            pdf_path=pdf_path,
            text="",
            images=[],
            total_pages=0,
            file_size_bytes=0,
            extraction_status="error",
            error_message=str(e)
        )


def _calculate_image_priority(pix: fitz.Pixmap, image_size: int) -> float:
    """
    Calculează un priority score pentru imagine (0.0 - 1.0).

    Folosit pentru a decide dacă imaginea trebuie processată cu OCR.
    Imagini cu text (diagrame, exerciții) primesc prioritate înaltă.
    Fotografi decorative primesc prioritate joasă.

    Args:
        pix: Pixmap object din PyMuPDF
        image_size: Mărimea imaginii în bytes

    Returns:
        Score între 0.0 (skip) și 1.0 (prioritate maximă)
    """

    score = 0.5  # Default score

    try:
        # Imagini mai mari probabil conțin mai mult text/diagramă
        # Maxim la ~500KB
        size_score = min(image_size / 512000, 1.0)

        # Aspect ratio: imagini pătrate/aproape pătrate sunt diagrame
        width = pix.width
        height = pix.height
        aspect_ratio = max(width, height) / max(min(width, height), 1)

        if 0.8 < aspect_ratio < 1.25:
            ratio_score = 0.9  # Aproape pătrate - probabil diagrame
        else:
            ratio_score = 0.5  # Landscape/portrait - fotograf

        # Combinează scores
        score = (size_score * 0.4) + (ratio_score * 0.6)

    except Exception as e:
        logger.debug(f"Error calculating priority score: {e}")

    return score


def _detect_image_type(pix: fitz.Pixmap) -> str:
    """
    Detectează tipul de imagine bazat pe caracteristici vizuale.

    Args:
        pix: Pixmap object

    Returns:
        Tipul imaginii: "diagram", "photo", "chart", "text", "unknown"
    """

    try:
        # Aspect ratio heuristic
        width = pix.width
        height = pix.height
        aspect_ratio = max(width, height) / max(min(width, height), 1)

        # Dimensiuni
        if width > 1000 or height > 1000:
            return "chart"  # Mare - probabil grafic
        elif width < 200 or height < 200:
            return "diagram"  # Mic - probabil diagramă
        elif 0.8 < aspect_ratio < 1.25:
            return "diagram"  # Aproape pătrat
        else:
            return "photo"  # Landscape/portrait

    except:
        return "unknown"


def get_pdf_metadata(pdf_path: str) -> Dict:
    """
    Extrage metadate din PDF (autor, titlu, etc).

    Args:
        pdf_path: Path la PDF

    Returns:
        Dict cu metadate disponibile
    """

    try:
        pdf_document = fitz.open(pdf_path)
        metadata = pdf_document.metadata
        pdf_document.close()

        return {
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "subject": metadata.get("subject", "Unknown"),
            "creator": metadata.get("creator", "Unknown")
        }
    except:
        return {
            "title": "Unknown",
            "author": "Unknown",
            "subject": "Unknown",
            "creator": "Unknown"
        }


# Exemplu de utilizare
if __name__ == "__main__":
    # Test pe un PDF mic din folderul de test
    test_pdf = Path(__file__).parent.parent / "materiale_didactice" / "test_sample.pdf"

    if test_pdf.exists():
        result = extract_text_and_images(str(test_pdf))
        print(f"Status: {result.extraction_status}")
        print(f"Text length: {len(result.text)} chars")
        print(f"Images found: {len(result.images)}")
        print(f"Pages: {result.total_pages}")
    else:
        print(f"Test PDF not found at {test_pdf}")
        print("Please add a PDF file to: materiale_didactice/test_sample.pdf")
