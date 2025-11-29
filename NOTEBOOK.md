# %% [markdown]
# # AI Educational - PDF to Supabase Pipeline (Production-Ready)
# 
# **Bazat pe modulele Python testate local de Edd**
# 
# - PyMuPDF (fitz) pentru extracție rapidă
# - PaddleOCR pentru imagini educaționale
# - Smart chunking cu MD5 deduplication
# - paraphrase-multilingual-mpnet-base-v2 (768 dim)
# - Batch upload optimizat Supabase
# 
# Structură input: `/kaggle/input/pdf-files/{clasa}/{materie}/file.pdf`

# %% [markdown]
# ## 1. Verificare Dependențe Pre-instalate Kaggle

# %%
import sys
import os
print(f"Python: {sys.version}")

# Check librării critice
dependencies_check = {
    'fitz (PyMuPDF)': False,
    'paddleocr': False,
    'sentence_transformers': False,
    'supabase': False,
    'tqdm': False
}

try:
    import fitz
    dependencies_check['fitz (PyMuPDF)'] = True
    print(f"✅ PyMuPDF: {fitz.__version__}")
except ImportError:
    print("❌ PyMuPDF not found")

try:
    import paddleocr
    dependencies_check['paddleocr'] = True
    print(f"✅ PaddleOCR installed")
except ImportError:
    print("❌ PaddleOCR not found")

try:
    from sentence_transformers import SentenceTransformer
    dependencies_check['sentence_transformers'] = True
    print(f"✅ sentence-transformers installed")
except ImportError:
    print("❌ sentence-transformers not found")

try:
    import supabase
    dependencies_check['supabase'] = True
    print(f"✅ supabase installed")
except ImportError:
    print("❌ supabase not found")

try:
    import tqdm
    dependencies_check['tqdm'] = True
    print(f"✅ tqdm installed")
except ImportError:
    print("❌ tqdm not found")

# %% [markdown]
# ## 2. Instalare Dependențe Lipsă

# %%
# Instalează DOAR ce lipsește
!pip install -q PyMuPDF==1.23.8
!pip install -q paddleocr==2.7.0.3
!pip install -q sentence-transformers==2.2.2
!pip install -q supabase==2.3.0
!pip install -q tqdm

print("✅ Toate dependențele instalate")

# %% [markdown]
# ## 3. Import-uri Principale

# %%
import warnings
warnings.filterwarnings('ignore')

import logging
import hashlib
import time
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

# PDF processing
import fitz  # PyMuPDF

# OCR
from paddleocr import PaddleOCR

# Embeddings
from sentence_transformers import SentenceTransformer

# Supabase
from supabase import create_client, Client

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("✅ Import-uri complete")

# %% [markdown]
# ## 4. Configurare Kaggle Secrets & Supabase

# %%
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

SUPABASE_URL = user_secrets.get_secret("SUPABASE_URL")
SUPABASE_KEY = user_secrets.get_secret("SUPABASE_KEY")

# Inițializare Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("✅ Supabase conectat")

# Test conexiune
try:
    response = supabase.table('document_embeddings').select('*').limit(1).execute()
    print(f"✅ Test query OK")
except Exception as e:
    print(f"⚠️  Test query failed: {e}")

# %% [markdown]
# ## 5. Configurare Modele (Embeddings + OCR)

# %%
# === EMBEDDING MODEL ===
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
EMBEDDING_DIM = 768

print(f"📥 Încărcare embedding model: {MODEL_NAME}")
embedding_model = SentenceTransformer(MODEL_NAME, device='cpu')

# Test
test_emb = embedding_model.encode(["Test text"])
assert len(test_emb[0]) == EMBEDDING_DIM
print(f"✅ Embedding model loaded: {EMBEDDING_DIM} dimensiuni")

# === OCR MODEL ===
print(f"📥 Încărcare PaddleOCR (ro + en)...")
ocr_model = PaddleOCR(
    use_angle_cls=True,
    lang=['ro', 'en'],
    use_gpu=False,  # CPU mai stabil pe Kaggle
    show_log=False
)
print(f"✅ PaddleOCR loaded")

# %% [markdown]
# ## 6. Clase & Funcții - PDF Extraction (PyMuPDF)

# %%
@dataclass
class ImageData:
    """Informații despre o imagine din PDF"""
    page_number: int
    bbox: Tuple[float, float, float, float]
    size_bytes: int
    priority_score: float
    image_type: str
    image_array: Optional[np.ndarray] = None  # Pentru OCR

@dataclass
class PDFExtractionResult:
    """Rezultat extracție PDF"""
    pdf_path: str
    text: str
    images: List[ImageData]
    total_pages: int
    file_size_bytes: int
    metadata: Dict
    extraction_status: str
    error_message: Optional[str] = None

def extract_metadata_from_path(pdf_path: str) -> Dict:
    """Extrage clasa și materia din structura de foldere"""
    path_parts = Path(pdf_path).parts
    
    metadata = {
        'clasa': None,
        'materie': None,
        'source_pdf': Path(pdf_path).name
    }
    
    # Căutare clasa
    for part in path_parts:
        if 'clasa' in part.lower() or 'class' in part.lower():
            match = re.search(r'\d+', part)
            if match:
                metadata['clasa'] = int(match.group())
                break
    
    # Căutare materie
    if len(path_parts) >= 2:
        metadata['materie'] = path_parts[-2]
    
    return metadata

def calculate_image_priority(pix: fitz.Pixmap, image_size: int) -> float:
    """Priority score 0.0-1.0 pentru OCR (imagini cu text = prioritate înaltă)"""
    score = 0.5
    
    try:
        size_score = min(image_size / 512000, 1.0)
        
        width = pix.width
        height = pix.height
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        
        if 0.8 < aspect_ratio < 1.25:
            ratio_score = 0.9  # Aproape pătrate - diagrame
        else:
            ratio_score = 0.5
        
        score = (size_score * 0.4) + (ratio_score * 0.6)
    except:
        pass
    
    return score

def detect_image_type(pix: fitz.Pixmap) -> str:
    """Detectează tipul imaginii"""
    try:
        width = pix.width
        height = pix.height
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        
        if width > 1000 or height > 1000:
            return "chart"
        elif width < 200 or height < 200:
            return "diagram"
        elif 0.8 < aspect_ratio < 1.25:
            return "diagram"
        else:
            return "photo"
    except:
        return "unknown"

def extract_text_and_images(pdf_path: str) -> PDFExtractionResult:
    """Extrage text și imagini din PDF cu PyMuPDF"""
    try:
        pdf_path = str(pdf_path)
        
        if not Path(pdf_path).exists():
            return PDFExtractionResult(
                pdf_path=pdf_path, text="", images=[], total_pages=0,
                file_size_bytes=0, metadata={}, extraction_status="error",
                error_message=f"File not found: {pdf_path}"
            )
        
        pdf_document = fitz.open(pdf_path)
        file_size_bytes = Path(pdf_path).stat().st_size
        total_pages = len(pdf_document)
        
        logger.info(f"Procesare: {Path(pdf_path).name} ({total_pages} pagini)")
        
        all_text = []
        images_found = []
        
        for page_number in range(total_pages):
            page = pdf_document[page_number]
            
            # Extrage text
            page_text = page.get_text()
            all_text.append(page_text)
            
            # Detectează imagini
            image_list = page.get_images()
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    image_bytes = pix.tobytes()
                    image_size = len(image_bytes)
                    
                    # Filtrează imagini mici (<50KB)
                    if image_size < 51200:
                        continue
                    
                    # Bounding box
                    image_rect = page.get_image_rects(img_info)
                    bbox = image_rect[0] if image_rect else (0, 0, 100, 100)
                    
                    priority = calculate_image_priority(pix, image_size)
                    
                    # Convertește pixmap la numpy array pentru OCR
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    
                    img_data = ImageData(
                        page_number=page_number + 1,
                        bbox=bbox,
                        size_bytes=image_size,
                        priority_score=priority,
                        image_type=detect_image_type(pix),
                        image_array=img_array
                    )
                    
                    images_found.append(img_data)
                    
                except Exception as e:
                    logger.warning(f"Skip image {img_index} p{page_number+1}: {e}")
                    continue
        
        combined_text = "\n".join(all_text)
        pdf_document.close()
        
        metadata = extract_metadata_from_path(pdf_path)
        
        logger.info(f"✅ Extras: {len(combined_text)} chars, {len(images_found)} imagini")
        
        return PDFExtractionResult(
            pdf_path=pdf_path,
            text=combined_text,
            images=images_found,
            total_pages=total_pages,
            file_size_bytes=file_size_bytes,
            metadata=metadata,
            extraction_status="success"
        )
        
    except Exception as e:
        logger.error(f"Eroare procesare {pdf_path}: {e}")
        return PDFExtractionResult(
            pdf_path=pdf_path, text="", images=[], total_pages=0,
            file_size_bytes=0, metadata={}, extraction_status="error",
            error_message=str(e)
        )

print("✅ PDF Extraction functions loaded")

# %% [markdown]
# ## 7. Funcții OCR (PaddleOCR)

# %%
def process_image_ocr(image_array: np.ndarray, min_confidence: float = 0.5) -> Tuple[str, float]:
    """
    Procesează imagine cu OCR și returnează (text, confidence)
    """
    try:
        ocr_result = ocr_model.ocr(image_array, cls=True)
        
        extracted_lines = []
        confidences = []
        
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                text = line[1][0]
                confidence = line[1][1]
                
                if confidence >= min_confidence:
                    extracted_lines.append(text)
                    confidences.append(confidence)
        
        full_text = "\n".join(extracted_lines)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return full_text, avg_confidence
        
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return "", 0.0

def process_images_selective(images: List[ImageData], priority_threshold: float = 0.7) -> List[Dict]:
    """
    Procesează doar imagini cu priority >= threshold
    """
    processed = []
    
    for img_data in images:
        if img_data.priority_score < priority_threshold:
            continue
        
        if img_data.image_array is None:
            continue
        
        ocr_text, confidence = process_image_ocr(img_data.image_array)
        
        if ocr_text.strip():
            processed.append({
                'page_number': img_data.page_number,
                'ocr_text': ocr_text,
                'confidence': confidence
            })
    
    return processed

print("✅ OCR functions loaded")

# %% [markdown]
# ## 8. Funcții Chunking cu Deduplication

# %%
@dataclass
class Chunk:
    """Chunk de text cu metadata"""
    text: str
    chunk_id: str
    chunk_hash: str
    source_page: int
    page_offset: int
    metadata: Dict

class TextChunker:
    """Smart chunker cu overlap și MD5 deduplication"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, min_chunk_length: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length
    
    def chunk_text(self, text: str, source_page: int = None, remove_duplicates: bool = True) -> List[Chunk]:
        """Split text în chunks inteligente"""
        if not text or not text.strip():
            return []
        
        text = self._clean_text(text)
        raw_chunks = self._split_chunks(text)
        
        chunk_objects = []
        seen_hashes: Set[str] = set()
        
        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text) < self.min_chunk_length:
                continue
            
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            if remove_duplicates and chunk_hash in seen_hashes:
                continue
            
            seen_hashes.add(chunk_hash)
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"chunk_{len(chunk_objects)}",
                chunk_hash=chunk_hash,
                source_page=source_page or 0,
                page_offset=i,
                metadata={
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split())
                }
            )
            
            chunk_objects.append(chunk)
        
        return chunk_objects
    
    def _split_chunks(self, text: str) -> List[str]:
        """Split text cu overlap"""
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk)
        
        if self.overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Adaugă overlap între chunks"""
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                prev_overlap = overlapped[i - 1][-self.overlap:]
                new_chunk = prev_overlap + " " + chunk
                overlapped.append(new_chunk)
        
        return overlapped
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split la punctuație"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        expanded = []
        for sent in sentences:
            parts = sent.split('\n')
            expanded.extend([p.strip() for p in parts if p.strip()])
        
        return expanded
    
    def _clean_text(self, text: str) -> str:
        """Curăță text"""
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        return text.strip()

print("✅ Chunking functions loaded")

# %% [markdown]
# ## 9. Funcții Embeddings

# %%
def generate_embeddings_batch(texts: List[str], batch_size: int = 128, show_progress: bool = True) -> List[np.ndarray]:
    """Generează embeddings pentru liste de texte"""
    if not texts:
        return []
    
    all_embeddings = []
    
    iterator = tqdm(range(0, len(texts), batch_size), desc="Embeddings") if show_progress else range(0, len(texts), batch_size)
    
    for i in iterator:
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, batch_size=len(batch), convert_to_numpy=True)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

print("✅ Embedding functions loaded")

# %% [markdown]
# ## 10. Funcții Supabase Upload

# %%
def upload_to_supabase(records: List[Dict], batch_size: int = 50, max_retries: int = 3) -> Dict:
    """Upload batch în Supabase cu retry logic"""
    success_count = 0
    failed_count = 0
    
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Upload Supabase")):
        # Format embeddings ca string
        formatted_batch = []
        for item in batch:
            embedding_str = '[' + ','.join(str(x) for x in item['embedding']) + ']'
            
            formatted_item = {
                'chunk_id': item['chunk_id'],
                'text': item['text'][:10000],
                'embedding': embedding_str,
                'source_pdf': item.get('source_pdf', 'unknown'),
                'page_num': int(item.get('page_num', 0)),
                'clasa': int(item.get('clasa', 0)) if item.get('clasa') else None,
                'materie': item.get('materie'),
                'capitol': item.get('capitol'),
                'chunk_hash': item.get('chunk_hash', ''),
                'has_images': bool(item.get('has_images', False))
            }
            formatted_batch.append(formatted_item)
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                supabase.table('document_embeddings').insert(formatted_batch).execute()
                success_count += len(batch)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    failed_count += len(batch)
                    logger.error(f"Batch {batch_idx} failed: {e}")
                else:
                    time.sleep(2)
    
    return {'success': success_count, 'failed': failed_count}

print("✅ Supabase upload functions loaded")

# %% [markdown]
# ## 11. Pipeline Principal - Procesare PDF Complet

# %%
def process_single_pdf(pdf_path: str, chunker: TextChunker, ocr_threshold: float = 0.7) -> List[Dict]:
    """
    Procesează un PDF complet: extract → OCR → chunk → embeddings
    
    Returns: Lista de records gata pentru Supabase
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Procesare: {Path(pdf_path).name}")
    logger.info(f"{'='*60}")
    
    # 1. Extracție PDF
    extraction_result = extract_text_and_images(pdf_path)
    
    if extraction_result.extraction_status != "success":
        logger.error(f"Extracție failed: {extraction_result.error_message}")
        return []
    
    # 2. OCR pe imagini high-priority
    ocr_texts = []
    if extraction_result.images:
        logger.info(f"  🔍 Procesare {len(extraction_result.images)} imagini (threshold={ocr_threshold})")
        ocr_results = process_images_selective(extraction_result.images, priority_threshold=ocr_threshold)
        ocr_texts = [r['ocr_text'] for r in ocr_results]
        logger.info(f"  ✅ OCR extras din {len(ocr_results)} imagini")
    
    # 3. Combină text PDF + OCR
    combined_text = extraction_result.text
    if ocr_texts:
        combined_text += "\n\n" + "\n\n".join(ocr_texts)
    
    if not combined_text.strip():
        logger.warning(f"  ⚠️  Niciun text extras")
        return []
    
    # 4. Chunking
    logger.info(f"  ✂️  Chunking text...")
    chunks = chunker.chunk_text(combined_text, source_page=1, remove_duplicates=True)
    logger.info(f"  ✅ {len(chunks)} chunks generate")
    
    if not chunks:
        return []
    
    # 5. Generare embeddings
    logger.info(f"  🧮 Generare embeddings...")
    chunk_texts = [c.text for c in chunks]
    embeddings = generate_embeddings_batch(chunk_texts, batch_size=128, show_progress=False)
    
    # 6. Construire records pentru Supabase
    metadata = extraction_result.metadata
    records = []
    
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        record = {
            'chunk_id': f"{metadata['source_pdf']}_{chunk.chunk_hash[:8]}_{idx}",
            'text': chunk.text,
            'embedding': embedding.tolist(),
            'source_pdf': metadata['source_pdf'],
            'page_num': chunk.source_page,
            'clasa': metadata.get('clasa'),
            'materie': metadata.get('materie'),
            'capitol': None,
            'chunk_hash': chunk.chunk_hash,
            'has_images': len(extraction_result.images) > 0
        }
        records.append(record)
    
    logger.info(f"  ✅ {len(records)} records pregătite pentru upload")
    return records

print("✅ Pipeline principal loaded")

# %% [markdown]
# ## 12. Checkpoint System

# %%
CHECKPOINT_FILE = '/kaggle/working/processing_checkpoint.pkl'

def save_checkpoint(processed_files: List[str], failed_files: List[str]):
    """Salvează progres"""
    checkpoint = {
        'processed': processed_files,
        'failed': failed_files,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"💾 Checkpoint: {len(processed_files)} procesate, {len(failed_files)} failed")

def load_checkpoint() -> Tuple[List[str], List[str]]:
    """Încarcă checkpoint"""
    if not os.path.exists(CHECKPOINT_FILE):
        return [], []
    
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        logger.info(f"📂 Checkpoint găsit: {checkpoint['timestamp']}")
        return checkpoint['processed'], checkpoint['failed']
    except:
        return [], []

print("✅ Checkpoint system loaded")

# %% [markdown]
# ## 13. Main Pipeline - Batch Processing

# %%
def find_all_pdfs(base_path: str) -> List[str]:
    """Găsește toate PDF-urile în structura de foldere"""
    pdf_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return sorted(pdf_files)

def main_pipeline(base_path: str, batch_upload_size: int = 50, checkpoint_interval: int = 3):
    """
    Pipeline complet de procesare PDF → Supabase
    """
    print("=" * 60)
    print("🚀 START PIPELINE - AI Educational")
    print("=" * 60)
    
    # Găsește PDFs
    all_pdfs = find_all_pdfs(base_path)
    print(f"\n📚 PDF-uri găsite: {len(all_pdfs)}")
    
    if not all_pdfs:
        print("❌ Niciun PDF găsit!")
        return
    
    # Încarcă checkpoint
    processed_files, failed_files = load_checkpoint()
    remaining_pdfs = [pdf for pdf in all_pdfs if pdf not in processed_files]
    
    print(f"✅ Deja procesate: {len(processed_files)}")
    print(f"⏳ De procesat: {len(remaining_pdfs)}")
    
    # Inițializare chunker
    chunker = TextChunker(chunk_size=500, overlap=50, min_chunk_length=50)
    
    # Stats
    total_chunks_uploaded = 0
    total_errors = 0
    
    # Procesare cu progress
    for idx, pdf_path in enumerate(tqdm(remaining_pdfs, desc="📄 Procesare PDF-uri")):
        try:
            # Procesează PDF
            records = process_single_pdf(pdf_path, chunker, ocr_threshold=0.7)
            
            if records:
                # Upload Supabase
                stats = upload_to_supabase(records, batch_size=batch_upload_size)
                total_chunks_uploaded += stats['success']
                total_errors += stats['failed']
                
                processed_files.append(pdf_path)
            else:
                failed_files.append(pdf_path)
        
        except Exception as e:
            logger.error(f"Eroare proces PDF {Path(pdf_path).name}: {e}")
            failed_files.append(pdf_path)
        
        # Checkpoint periodic
        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(processed_files, failed_files)
    
    # Checkpoint final
    save_checkpoint(processed_files, failed_files)
    
    # Stats finale
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLET")
    print("=" * 60)
    print(f"📄 PDF-uri procesate: {len(processed_files)}")
    print(f"❌ PDF-uri failed: {len(failed_files)}")
    print(f"📦 Total chunks uploaded: {total_chunks_uploaded}")
    print(f"⚠️  Total erori upload: {total_errors}")
    
    # Stats Supabase
    try:
        response = supabase.table('document_embeddings').select('count', count='exact').execute()
        print(f"\n📊 Total vectors în Supabase: {response.count:,}")
    except Exception as e:
        print(f"⚠️  Nu pot accesa stats: {e}")

print("✅ Main pipeline loaded")

# %% [markdown]
# ## 14. TEST MODE - Procesare 1 PDF

# %%
# Configurare
PDF_BASE_PATH = '/kaggle/input/pdf-files'
TEST_MODE = True

if TEST_MODE:
    print("🧪 TEST MODE - procesare 1 PDF\n")
    
    all_pdfs = find_all_pdfs(PDF_BASE_PATH)
    
    if all_pdfs:
        test_pdf = all_pdfs[0]
        print(f"Test PDF: {test_pdf}\n")
        
        chunker = TextChunker(chunk_size=500, overlap=50)
        records = process_single_pdf(test_pdf, chunker, ocr_threshold=0.7)
        
        if records:
            print(f"\n✅ {len(records)} records generate")
            print(f"\n📦 Exemplu record:")
            example = {k: v if k != 'embedding' else f"[{len(v)} dims]" for k, v in records[0].items()}
            print(json.dumps(example, indent=2, ensure_ascii=False))
            
            # Test upload 1 record
            print(f"\n🧪 Test upload Supabase (primul record)...")
            stats = upload_to_supabase([records[0]], batch_size=1)
            
            if stats['success'] >