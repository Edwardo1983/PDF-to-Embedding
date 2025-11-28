# 🐛 TROUBLESHOOTING.md - Solutions for Common Issues

Ghid pentru rezolvarea problemelor comune în pipeline PDF → Embeddings.

---

## 🔴 KAGGLE ISSUES

### ❌ "GPU Not Available" / "Accelerator: None"

**Symptom:**
```
No GPU selected in notebook
Accelerator dropdown shows only "None"
!nvidia-smi returns "command not found"
```

**Causes:**
- Phone verification not completed
- Phone number rejected by Kaggle
- Account age < 5 days (Kaggle restriction)

**Solution:**

1. **Verify phone:**
   ```
   Dashboard → Settings → Phone
   Status should show: ✅ Verified
   ```

2. **If not verified, try:**
   - Use different phone number (mobile preferred)
   - Wait 24 hours after account creation
   - Contact Kaggle support if still fails

3. **Once verified:**
   - Create new notebook
   - Settings → Accelerator → GPU
   - Restart kernel
   - Run: `!nvidia-smi`
   - Should show: Tesla P100, 16GB memory

**If still fails:**
- Try different browser (Chrome vs Firefox)
- Clear cookies: Settings → Clear browsing data
- Create new notebook from scratch

---

### ❌ "Out of Memory" / "CUDA Out of Memory"

**Symptom:**
```
RuntimeError: CUDA out of memory
Killed (OOM killer)
Notebook session terminates
```

**Cause:** Batch size too large for available GPU memory

**Solution:**

1. **Reduce batch size:**
   ```yaml
   # config.yaml
   embeddings:
     batch_size: 128  # ← Change to:
     batch_size: 64   # (or even 32)
   ```

2. **Reduce other processes:**
   ```python
   # In notebook, clear memory
   import gc
   gc.collect()  # Force garbage collection

   # Don't load multiple large models simultaneously
   ```

3. **Process in chunks:**
   ```python
   # Instead of all PDFs at once:
   all_pdfs = get_all_pdfs()  # 5000 PDFs

   # Process in groups of 100
   for i in range(0, len(all_pdfs), 100):
       batch_pdfs = all_pdfs[i:i+100]
       # Process and upload immediately
       upload_vectors(process(batch_pdfs))
   ```

4. **Monitor memory:**
   ```python
   # Add to notebook
   import psutil
   print(f"Memory: {psutil.virtual_memory().percent}%")
   print(f"GPU memory: check with !nvidia-smi")
   ```

---

### ❌ "Notebook Timeout" / "Session Disconnected"

**Symptom:**
```
Notebook stops responding after 6-8 hours
Output shows: "Kernel dead" or "Connection lost"
```

**Causes:**
- Kaggle notebook timeout (max 9 hours session)
- Inactivity (no output for >1 hour)
- Network interruption

**Solution:**

1. **Split processing into multiple runs:**
   ```python
   # If you have 5000 PDFs and processing takes 20+ hours:
   # Split into 3 notebooks:
   # - Notebook 1: Process PDFs 0-1800
   # - Notebook 2: Process PDFs 1800-3600
   # - Notebook 3: Process PDFs 3600-5000
   ```

2. **Use checkpoints:**
   ```python
   # After every 50 PDFs, upload to Supabase immediately
   if pdf_count % 50 == 0:
       upload_vectors_batch(current_vectors)
       print(f"Checkpoint: uploaded {len(current_vectors)} vectors")
       current_vectors = []  # Clear for next batch
   ```

3. **Keep notebook active:**
   ```python
   # Add output to show progress (prevents timeout)
   from tqdm import tqdm
   import time

   for pdf in tqdm(pdfs):
       # ... processing ...
       # Progress bar output keeps notebook "alive"
   ```

---

### ❌ "Dataset Not Found" / "No Input Files"

**Symptom:**
```
FileNotFoundError: /kaggle/input/dataset not found
os.listdir('/kaggle/input/') returns empty or different folder
```

**Cause:** Dataset not attached or named incorrectly

**Solution:**

1. **Check attached datasets:**
   ```python
   import os
   print(os.listdir('/kaggle/input/'))
   # Should show: ['materiale-didactice-0-4', ...]
   ```

2. **If empty or wrong dataset:**
   - Click "+" next to "Input" (right sidebar)
   - Search your dataset by name
   - Click to attach
   - Restart notebook kernel

3. **Update notebook code:**
   ```python
   # Check your actual dataset folder name
   dataset_name = os.listdir('/kaggle/input/')[0]
   pdf_folder = f'/kaggle/input/{dataset_name}/'
   print(os.listdir(pdf_folder))  # List files
   ```

---

### ❌ "GPU P100 Hours Exceeded"

**Symptom:**
```
Accelerator not available
Message: "You've used 30 hours this week"
```

**Causes:** Weekly quota exceeded (Kaggle limit: 30h/week)

**Solution:**

1. **Wait for weekly reset** (every Sunday UTC)

2. **In the meantime:**
   - Process on CPU (slower but free unlimited)
   - Set `use_gpu: false` in config
   - Reduce batch_size to 32
   - Takes 2-3x longer but still works

3. **Alternative:**
   - Use Google Colab (12h free/day, GPU available)
   - Upload script to Colab
   - Run there while waiting for Kaggle reset

---

## 🟠 SUPABASE ISSUES

### ❌ "Connection Timeout" / "Failed to Connect"

**Symptom:**
```
supabase.ConnectionError: Failed to connect to https://xxxxx.supabase.co
Timeout after 30 seconds
```

**Causes:**
- Invalid API key
- Wrong project URL
- Network connectivity issue
- Supabase project not running

**Solution:**

1. **Verify credentials:**
   ```python
   # Check secrets in Kaggle notebook
   from kaggle_secrets import UserSecretsClient
   secret = UserSecretsClient()

   url = secret.get_secret('SUPABASE_URL')
   key = secret.get_secret('SUPABASE_ANON_KEY')

   print(f"URL: {url}")
   print(f"Key (first 20 chars): {key[:20]}")

   # Should print: https://xxxxx.supabase.co
   # Key should start with: eyJhbGciOi...
   ```

2. **Check Supabase project status:**
   - Go to [supabase.com/dashboard](https://supabase.com/dashboard)
   - Click your project
   - Status should show: "Running"
   - If paused: click "Resume" button

3. **Test connection simply:**
   ```python
   import requests
   url = "https://xxxxx.supabase.co/rest/v1/"
   response = requests.get(url, headers={"apikey": key})
   print(response.status_code)  # Should be 200 or 401 (not timeout)
   ```

4. **Check network:**
   ```python
   # If in Kaggle, check internet access
   !curl -I https://google.com
   # Should return: HTTP/1.1 200 OK
   ```

---

### ❌ "pgvector Extension Not Enabled"

**Symptom:**
```
Error: column embedding type "vector" does not exist
HINT: No operator matches the given name and argument type(s)
```

**Cause:** pgvector extension not created on Supabase

**Solution:**

1. **Enable extension:**
   ```
   Supabase Dashboard → SQL Editor → New Query
   ```

2. **Run:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Verify:**
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   # Should return 1 row
   ```

4. **If fails:**
   - Wait 1-2 minutes (Supabase might be updating)
   - Try again
   - If still fails: check project status (might be provisioning)

---

### ❌ "Vectors Not Uploading" / "Insert Failed"

**Symptom:**
```
psycopg2.errors.CheckViolation: new row for relation document_embeddings
violates check constraint
```

**Causes:**
- Embedding dimension mismatch (not exactly 768)
- NULL values in required columns
- Chunk_id already exists (duplicate)

**Solution:**

1. **Validate vectors before upload:**
   ```python
   def validate_vectors(vectors):
       for vec in vectors:
           # Check dimensions
           if len(vec['embedding']) != 768:
               raise ValueError(f"Vector dimension {len(vec['embedding'])} != 768")

           # Check required fields
           required = ['chunk_id', 'text', 'embedding', 'source_pdf', 'clasa']
           for req in required:
               if req not in vec or vec[req] is None:
                   raise ValueError(f"Missing required field: {req}")

       return True
   ```

2. **Check for duplicates:**
   ```python
   # Ensure chunk_id is unique
   chunk_ids = [v['chunk_id'] for v in vectors]
   if len(chunk_ids) != len(set(chunk_ids)):
       print("Duplicate chunk_ids found!")
       # Check what's duplicate
       seen = set()
       for cid in chunk_ids:
           if cid in seen:
               print(f"Duplicate: {cid}")
           seen.add(cid)
   ```

3. **Test insert small batch:**
   ```python
   # First, test with just 1 vector
   test_vector = [vectors[0]]
   result = uploader.upload_vectors(test_vector)

   if result['failed'] > 0:
       print("Small batch failed - debug before uploading all")
   else:
       print("Small batch OK - proceed with full upload")
   ```

---

### ❌ "HNSW Index Creation Slow" / "Index Timeout"

**Symptom:**
```
CREATE INDEX takes 1+ hours
Query: "CREATE INDEX ... USING hnsw ..." still running
```

**Status:** This is **NORMAL** for 600k+ vectors!

**Expected times:**
- 100k vectors: ~5-10 minutes
- 300k vectors: ~20-30 minutes
- 600k vectors: ~60-90 minutes

**Solution:**

1. **Just wait.** Index creation is a one-time cost.

2. **Monitor progress:**
   ```sql
   -- Check if index is still building
   SELECT schemaname, tablename, indexname
   FROM pg_indexes
   WHERE tablename = 'document_embeddings';

   -- Will show index as soon as it exists
   -- If not in results, still building
   ```

3. **If you cancel:**
   ```sql
   -- Don't cancel! But if you did by accident:
   DROP INDEX IF EXISTS idx_embedding_hnsw;

   -- Recreate (will take just as long)
   CREATE INDEX idx_embedding_hnsw
   ON document_embeddings
   USING hnsw (embedding vector_cosine_ops);
   ```

---

## 🟡 OCR ISSUES

### ❌ "PaddleOCR Model Download Very Slow"

**Symptom:**
```
Downloading paddleocr model... (first run takes 10-15 min)
Network seems stuck
```

**Status:** Normal. Models are ~500MB each.

**Solution:**

1. **Just wait.** First initialization downloads models (~10-15 min for CPU models)

2. **Check progress:**
   ```python
   # Monitor in notebook output
   # Should see download progress
   [***  ] 34% downloaded
   ```

3. **If stuck (no progress for 5+ min):**
   - Restart notebook (interrupt kernel)
   - Try again (might retry partially downloaded)
   - If still fails, use CPU (default, usually works)

---

### ❌ "OCR Confidence Too Low"

**Symptom:**
```
OCR results: "confidence: 0.23"
Extracted text is gibberish
```

**Cause:** Image quality poor or not text (e.g., photo)

**Solution:**

1. **Increase priority threshold:**
   ```yaml
   # config.yaml
   ocr:
     priority_threshold: 0.9  # Only process high-quality diagrams
   ```

2. **Pre-process images:**
   ```python
   # (Optional) Enhance image before OCR
   import cv2
   import numpy as np

   # Enhance contrast
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   enhanced = clahe.apply(image)

   result = ocr.ocr(enhanced)  # Better results
   ```

3. **Accept low confidence gracefully:**
   ```python
   # In ocr_processor.py, filter by confidence
   if result.confidence < 0.5:
       # Skip this result
       continue
   ```

---

## 🔵 PDF EXTRACTION ISSUES

### ❌ "PDF Parsing Fails" / "Cannot Extract Text"

**Symptom:**
```
fitz.open() raises exception
ValueError: PDF file is encrypted
```

**Cause:** Corrupted or encrypted PDF

**Solution:**

1. **Graceful error handling (already in code):**
   ```python
   # In pdf_extractor.py - wraps with try/except
   try:
       result = extract_text_and_images(pdf_path)
   except Exception as e:
       logger.warning(f"Skipped {pdf_path}: {e}")
       continue

   # Result will show:
   # ✅ Processed: 4950
   # ⚠️ Skipped (error): 50
   ```

2. **Check specific PDF:**
   ```python
   import fitz
   try:
       pdf = fitz.open("problem.pdf")
       text = pdf[0].get_text()
       print(f"Extracted: {text[:100]}")
   except Exception as e:
       print(f"Error: {e}")
       print("PDF might be corrupted or encrypted")
   ```

3. **If encrypted:**
   ```python
   # Check if PDF is encrypted
   pdf = fitz.open("problem.pdf")
   if pdf.is_encrypted:
       # Try to unlock (if no password)
       pdf.load_page(0)  # Might fail if password required
   ```

---

### ❌ "All Text on One Line" / "Formatting Lost"

**Symptom:**
```
PDF text: "This is first sentence. This is second sentence."
Expected: "This is first sentence.\nThis is second sentence."
```

**Cause:** PDF doesn't have explicit line breaks

**Status:** Acceptable - semantic chunking doesn't require perfect formatting

**If you need better formatting:**
```python
# Split by periods/punctuation (manual improvement)
import re

text = pdf_text
text = re.sub(r'\.(?=[A-Z])', '.\n', text)  # Add newline after period
text = re.sub(r'\. ', '.\n', text)  # Period + space → newline

# Then chunk
```

---

## 🟢 EMBEDDING ISSUES

### ❌ "Embedding Generation Very Slow"

**Symptom:**
```
Processing 600k chunks takes 30+ hours
Rate: ~500 vectors/hour (too slow)
```

**Cause:** Batch size too small or CPU processing

**Solution:**

1. **Increase batch size (if GPU available):**
   ```yaml
   embeddings:
     batch_size: 256  # (was 128)
   ```

2. **Use GPU if available:**
   ```yaml
   embeddings:
     device: "cuda"  # if GPU available
   ```

3. **On CPU, 500 vectors/hour is expected:**
   - Kaggle CPU: ~100-200 vectors/hour
   - GPU: 50,000+ vectors/hour
   - Total for 600k: 6-12 hours (GPU) vs 50+ hours (CPU)
   - GPU processing is CRITICAL for timeline

---

### ❌ "Embedding Dimension Mismatch"

**Symptom:**
```
ValueError: embedding shape (1, 768) != expected (1, 384)
```

**Cause:** Wrong model or dimension setting

**Solution:**

1. **Check model output:**
   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
   embedding = model.encode(['test'])
   print(embedding.shape)  # Should be (1, 768)
   ```

2. **Verify config:**
   ```yaml
   embeddings:
     dimensions: 768  # Match model output
   ```

3. **Check database schema:**
   ```sql
   -- Should be VECTOR(768), not VECTOR(384)
   SELECT column_name, data_type
   FROM information_schema.columns
   WHERE table_name = 'document_embeddings' AND column_name = 'embedding';
   ```

---

## 🟣 GENERAL DEBUGGING

### How to Enable Debug Logging

```python
# In notebook, set logging level
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in code
logger.setLevel(logging.DEBUG)

# Now see detailed debug messages
```

### Common Debug Commands

```python
# Check file counts
import os
pdf_folder = '/kaggle/input/dataset/'
all_pdfs = []
for root, dirs, files in os.walk(pdf_folder):
    for file in files:
        if file.endswith('.pdf'):
            all_pdfs.append(os.path.join(root, file))

print(f"Total PDFs: {len(all_pdfs)}")
print(f"Sample PDFs: {all_pdfs[:3]}")

# Check Supabase connection
from supabase import create_client
supabase = create_client(url, key)
response = supabase.table('document_embeddings').select("count", count="exact").execute()
print(f"Vectors in DB: {response.count}")

# Check memory usage
import psutil
print(f"RAM used: {psutil.virtual_memory().percent}%")
print(f"Available: {psutil.virtual_memory().available / 1e9:.1f} GB")
```

---

## 📞 Getting Help

If issue not resolved:

1. **Check logs:**
   - Kaggle notebook output
   - `logs/processing.log` (if created)
   - Supabase error messages

2. **Search similar issues:**
   - GitHub: pdf-to-embedding issues
   - Stack Overflow: search error message
   - Kaggle Discussions: search issue

3. **Create detailed bug report:**
   ```
   Title: [Component] Issue description

   Steps to reproduce:
   1. ...
   2. ...

   Expected behavior: ...
   Actual behavior: ...

   Error message:
   [Full error traceback]

   Environment:
   - Python version
   - OS (Windows/Linux/Mac)
   - Kaggle GPU/CPU
   ```

---

**Updated: 2024**
**Component Versions: PyMuPDF 1.23+, PaddleOCR 2.7+, sentence-transformers 2.2+**
