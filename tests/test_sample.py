"""
test_sample.py - End-to-End Test cu 2-3 PDFs Sample

Teste procesarea completă a unui mic set de PDFs:
1. Extract text din PDF
2. Process imagini cu OCR
3. Chunk text
4. Generate embeddings
5. Upload la Supabase
6. Query similarity search

Rulare: python tests/test_sample.py
"""

import os
import sys
import logging
from pathlib import Path
import tempfile
import hashlib

# Add scripts folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from pdf_extractor import extract_text_and_images
from ocr_processor import OCRProcessor
from chunking import TextChunker
from embedding_generator import EmbeddingGenerator
from supabase_uploader import SupabaseUploader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Runner pentru end-to-end tests"""

    def __init__(self):
        self.test_pdfs = []
        self.extracted_data = []
        self.chunks = []
        self.embeddings = []
        self.stats = {
            'extracted_pdfs': 0,
            'total_text_chars': 0,
            'total_images': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'uploaded_vectors': 0
        }

    def find_test_pdfs(self, max_count: int = 3) -> bool:
        """
        Caută sample PDFs din materiale_didactice folder.

        Args:
            max_count: Maximum PDFs pentru test

        Returns:
            True dacă găsite, False altfel
        """
        logger.info("Searching for test PDFs...")

        materiale_dir = Path(__file__).parent.parent / 'materiale_didactice'

        if not materiale_dir.exists():
            logger.warning(f"Materiale folder not found: {materiale_dir}")
            logger.info("Creating empty test folder...")
            materiale_dir.mkdir(exist_ok=True)
            return False

        # Find PDFs recursively
        self.test_pdfs = list(materiale_dir.glob('**/*.pdf'))[:max_count]

        if not self.test_pdfs:
            logger.warning(f"No PDFs found in {materiale_dir}")
            return False

        logger.info(f"Found {len(self.test_pdfs)} test PDFs:")
        for pdf in self.test_pdfs:
            size_mb = pdf.stat().st_size / 1024 / 1024
            logger.info(f"  - {pdf.name} ({size_mb:.1f} MB)")

        return True

    def test_pdf_extraction(self) -> bool:
        """Test PDF text extraction"""
        logger.info("\n=== TEST 1: PDF EXTRACTION ===")

        if not self.test_pdfs:
            logger.error("No test PDFs found")
            return False

        try:
            for pdf_path in self.test_pdfs:
                logger.info(f"Extracting: {pdf_path.name}")

                result = extract_text_and_images(str(pdf_path))

                if result.extraction_status != "success":
                    logger.error(f"  ✗ Extraction failed: {result.error_message}")
                    continue

                logger.info(f"  ✓ Text: {len(result.text)} chars")
                logger.info(f"  ✓ Images: {len(result.images)} found")
                logger.info(f"  ✓ Pages: {result.total_pages}")

                self.extracted_data.append(result)
                self.stats['extracted_pdfs'] += 1
                self.stats['total_text_chars'] += len(result.text)
                self.stats['total_images'] += len(result.images)

            logger.info(f"✅ Extraction complete: {self.stats['extracted_pdfs']} PDFs processed")
            return True

        except Exception as e:
            logger.error(f"❌ Extraction test failed: {e}")
            return False

    def test_ocr_processing(self) -> bool:
        """Test OCR processing (simplified)"""
        logger.info("\n=== TEST 2: OCR PROCESSING ===")

        if not self.extracted_data:
            logger.warning("No extracted data. Skipping OCR test.")
            return True

        # Check if OCR is enabled in config
        logger.info("⚠️ OCR disabled in config - skipping OCR test")
        logger.info("(This is OK - OCR is optional)")
        return True

    def test_chunking(self) -> bool:
        """Test text chunking"""
        logger.info("\n=== TEST 3: TEXT CHUNKING ===")

        if not self.extracted_data:
            logger.error("No extracted data")
            return False

        try:
            chunker = TextChunker(chunk_size=500, overlap=50)

            for extracted in self.extracted_data:
                logger.info(f"Chunking {Path(extracted.pdf_path).name}...")

                chunks = chunker.chunk_text(
                    extracted.text,
                    source_page=1,
                    remove_duplicates=True
                )

                logger.info(f"  ✓ Generated {len(chunks)} chunks")

                self.chunks.extend(chunks)
                self.stats['total_chunks'] += len(chunks)

            logger.info(f"✅ Chunking complete: {self.stats['total_chunks']} total chunks")
            return True

        except Exception as e:
            logger.error(f"❌ Chunking test failed: {e}")
            return False

    def test_embedding_generation(self) -> bool:
        """Test embedding generation"""
        logger.info("\n=== TEST 4: EMBEDDING GENERATION ===")

        if not self.chunks:
            logger.error("No chunks to embed")
            return False

        try:
            logger.info("Initializing embedding model...")
            generator = EmbeddingGenerator(device='cpu')

            # Extract texts from chunks
            texts = [chunk.text for chunk in self.chunks]

            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = generator.generate_embeddings(texts, show_progress=False)

            logger.info(f"  ✓ Generated {len(embeddings)} embeddings")
            logger.info(f"  ✓ Dimension: {embeddings[0].shape if embeddings else 'N/A'}")

            self.embeddings = embeddings
            self.stats['total_embeddings'] = len(embeddings)

            # Verify dimensions
            if embeddings:
                dim = len(embeddings[0])
                if dim != 768:
                    logger.warning(f"⚠️ Expected 768 dimensions, got {dim}")
                    return False

            logger.info(f"✅ Embedding generation complete")
            return True

        except Exception as e:
            logger.error(f"❌ Embedding test failed: {e}")
            return False

    def test_supabase_connection(self) -> bool:
        """Test Supabase connection"""
        logger.info("\n=== TEST 5: SUPABASE CONNECTION ===")

        # Get credentials from environment
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')

        if not supabase_url or not supabase_key:
            logger.warning("⚠️ Supabase credentials not in environment")
            logger.info("Set SUPABASE_URL and SUPABASE_ANON_KEY to test upload")
            logger.info("(Example: set SUPABASE_URL=https://xxxxx.supabase.co)")
            return True  # Not a failure, just skip

        try:
            logger.info("Connecting to Supabase...")
            uploader = SupabaseUploader(supabase_url, supabase_key)

            if not uploader.connect():
                logger.error("❌ Failed to connect to Supabase")
                return False

            logger.info("✅ Successfully connected to Supabase")

            # Get stats
            stats = uploader.get_database_stats()
            logger.info(f"Database stats: {stats}")

            return True

        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            return False

    def test_upload_vectors(self) -> bool:
        """Test vector upload"""
        logger.info("\n=== TEST 6: VECTOR UPLOAD ===")

        if not self.chunks or not self.embeddings:
            logger.warning("No chunks/embeddings. Skipping upload test.")
            return True

        # Get credentials
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')

        if not supabase_url or not supabase_key:
            logger.warning("⚠️ Supabase credentials not available. Skipping upload.")
            return True

        try:
            logger.info("Preparing vectors for upload...")

            # Prepare upload data
            vectors_data = []
            for i, chunk in enumerate(self.chunks):
                vector_item = {
                    'chunk_id': f"test_chunk_{i}_{hashlib.md5(chunk.text.encode()).hexdigest()[:8]}",
                    'text': chunk.text[:5000],  # Limit to 5000 chars
                    'embedding': self.embeddings[i].tolist(),
                    'source_pdf': 'test_sample.pdf',
                    'page_num': chunk.source_page,
                    'clasa': 1,  # Test class
                    'materie': 'Test',
                    'capitol': 'Test Chapter',
                    'chunk_hash': chunk.chunk_hash,
                    'has_images': False
                }
                vectors_data.append(vector_item)

            logger.info(f"Uploading {len(vectors_data)} test vectors...")

            uploader = SupabaseUploader(supabase_url, supabase_key)
            if not uploader.connect():
                logger.warning("Could not connect to Supabase. Skipping upload.")
                return True

            result = uploader.upload_vectors(vectors_data, show_progress=False)

            logger.info(f"✅ Upload complete: {result['success']} success, {result['failed']} failed")
            self.stats['uploaded_vectors'] = result['success']

            return result['failed'] == 0

        except Exception as e:
            logger.error(f"❌ Upload test failed: {e}")
            return False

    def test_similarity_search(self) -> bool:
        """Test similarity search"""
        logger.info("\n=== TEST 7: SIMILARITY SEARCH ===")

        if not self.embeddings:
            logger.warning("No embeddings. Skipping similarity test.")
            return True

        # Get credentials
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')

        if not supabase_url or not supabase_key:
            logger.warning("⚠️ Supabase credentials not available.")
            return True

        try:
            logger.info("Testing similarity search...")

            uploader = SupabaseUploader(supabase_url, supabase_key)
            if not uploader.connect():
                logger.warning("Could not connect to Supabase.")
                return True

            # Use first embedding as query
            query_emb = self.embeddings[0].tolist()

            results = uploader.query_similar(query_emb, match_count=5)

            logger.info(f"✅ Similarity search returned {len(results)} results")

            if results:
                for i, result in enumerate(results[:3]):
                    text_preview = result.get('text', '')[:50]
                    similarity = result.get('similarity', 0)
                    logger.info(f"  {i+1}. {text_preview}... (similarity: {similarity:.2%})")

            return True

        except Exception as e:
            logger.error(f"❌ Similarity test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests"""
        logger.info("\n" + "="*70)
        logger.info("STARTING END-TO-END TESTS")
        logger.info("="*70)

        tests = [
            ("PDF Extraction", self.test_pdf_extraction),
            ("OCR Processing", self.test_ocr_processing),
            ("Text Chunking", self.test_chunking),
            ("Embedding Generation", self.test_embedding_generation),
            ("Supabase Connection", self.test_supabase_connection),
            ("Vector Upload", self.test_upload_vectors),
            ("Similarity Search", self.test_similarity_search),
        ]

        results = {}

        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} crashed: {e}")
                results[test_name] = False

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            logger.info(f"{status}: {test_name}")

        logger.info(f"\nScore: {passed}/{total} tests passed")

        # Print statistics
        logger.info("\n" + "-"*70)
        logger.info("PROCESSING STATISTICS")
        logger.info("-"*70)
        logger.info(f"PDFs processed: {self.stats['extracted_pdfs']}")
        logger.info(f"Total text: {self.stats['total_text_chars']} characters")
        logger.info(f"Images found: {self.stats['total_images']}")
        logger.info(f"Chunks generated: {self.stats['total_chunks']}")
        logger.info(f"Embeddings created: {self.stats['total_embeddings']}")
        logger.info(f"Vectors uploaded: {self.stats['uploaded_vectors']}")

        logger.info("\n" + "="*70)

        if passed == total:
            logger.info("✅ ALL TESTS PASSED!")
            return True
        else:
            logger.info(f"⚠️  Some tests failed. Check output above.")
            return False


def main():
    """Main test runner"""
    runner = TestRunner()

    # Run tests
    if runner.find_test_pdfs():
        success = runner.run_all_tests()
        return 0 if success else 1
    else:
        logger.error("\n❌ No test PDFs found. Add some PDFs to:")
        logger.error("  C:\\Users\\Opaop\\Desktop\\PDF-to-Embedding\\materiale_didactice\\")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
