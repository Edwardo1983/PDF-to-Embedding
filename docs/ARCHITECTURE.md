# 🏗️ ARCHITECTURE.md - Explicație Tehnică Complete

Documentație detaliată a arhitecturii pipeline-ului PDF → Embeddings → Supabase.

---

## 📊 Flow Diagram Complet

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: 15GB PDF-uri LOCAL                    │
│        [C:\materiale_didactice\clasa_X\materie\*.pdf]           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─── [MANUAL UPLOAD TO KAGGLE]
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│             KAGGLE DATASET (mounted as /kaggle/input)           │
│  - Full 15GB folder structure preserved                         │
│  - ~3,000-5,000 PDF files                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─── [KAGGLE NOTEBOOK PROCESSING]
                     │
                     ↓
        ╔════════════════════════════════════════╗
        ║  PROCESSING PIPELINE (Kaggle P100)    ║
        ╠════════════════════════════════════════╣
        │                                        │
        │  ┌──────────────────────────────────┐ │
        │  │1. PDF EXTRACTION (PyMuPDF)       │ │
        │  ├──────────────────────────────────┤ │
        │  │ Input:  PDF file                 │ │
        │  │ Extract:                         │ │
        │  │  - Text from all pages           │ │
        │  │  - Images (>50KB)                │ │
        │  │  - Page boundaries               │ │
        │  │ Output: {text, images[], meta}   │ │
        │  │ Speed: ~500 PDFs/hour (GPU acc)  │ │
        │  └──────────────────────────────────┘ │
        │           │                            │
        │           ├─ Text (~80%)               │
        │           └─ Images (~20%)              │
        │                                        │
        │  ┌──────────────────────────────────┐ │
        │  │2. OCR PROCESSING (PaddleOCR)     │ │
        │  ├──────────────────────────────────┤ │
        │  │ Input:  Images from PDFs         │ │
        │  │ Process:                         │ │
        │  │  - Score images (priority)       │ │
        │  │  - Filter low-priority ones      │ │
        │  │  - Run OCR на selected images    │ │
        │  │ Output: Extracted text + conf    │ │
        │  │ Speed: ~1,000 images/hour        │ │
        │  └──────────────────────────────────┘ │
        │           │                            │
        │           └─ OCR text merged with PDF  │
        │                                        │
        │  ┌──────────────────────────────────┐ │
        │  │3. TEXT CHUNKING & DEDUP          │ │
        │  ├──────────────────────────────────┤ │
        │  │ Input:  Full text from PDF       │ │
        │  │ Process:                         │ │
        │  │  - Split at sentence boundaries  │ │
        │  │  - Target: ~500 chars/chunk      │ │
        │  │  - Add 50-char overlap           │ │
        │  │  - Calculate MD5 hashes          │ │
        │  │  - Remove duplicates (100 chars) │ │
        │  │ Output: [Chunk objects]          │ │
        │  │ Example:                         │ │
        │  │  Chunk1: "Aria pătrat = ... "    │ │
        │  │  Chunk2: "... = latura² "        │ │
        │  │  (Overlap: last 50 chars chk1)   │ │
        │  └──────────────────────────────────┘ │
        │           │                            │
        │           └─ Dedup rate: ~2-5%         │
        │                                        │
        │  ┌──────────────────────────────────┐ │
        │  │4. EMBEDDING GENERATION           │ │
        │  ├──────────────────────────────────┤ │
        │  │ Model:                           │ │
        │  │  paraphrase-multilingual-        │ │
        │  │  mpnet-base-v2                   │ │
        │  │                                  │ │
        │  │ Input:  Chunks [str]             │ │
        │  │ Process:                         │ │
        │  │  - Batch size: 128 texts        │ │
        │  │  - Tokenize (max 384 tokens)     │ │
        │  │  - Forward pass (GPU)            │ │
        │  │  - Extract pooled output         │ │
        │  │  - Output: 768-dim vector        │ │
        │  │ Output: numpy arrays (N, 768)    │ │
        │  │ Speed: ~50,000 vecs/hour (GPU)   │ │
        │  └──────────────────────────────────┘ │
        │           │                            │
        │           └─ One vector = semantic     │
        │             representation            │
        │                                        │
        │  ┌──────────────────────────────────┐ │
        │  │5. BATCH UPLOAD TO SUPABASE       │ │
        │  ├──────────────────────────────────┤ │
        │  │ Input:  Vectors + metadata       │ │
        │  │ Process:                         │ │
        │  │  - Group: 10,000 vectors/batch   │ │
        │  │  - Format for pgvector           │ │
        │  │  - Prepare metadata JSON         │ │
        │  │  - POST to Supabase REST API     │ │
        │  │  - Retry logic (3 attempts)      │ │
        │  │  - Wait between batches          │ │
        │  │ Output: Vectors in DB            │ │
        │  │ Speed: ~10,000/min (batched)     │ │
        │  │ Total time: 2-3 hours (600k)     │ │
        │  └──────────────────────────────────┘ │
        │                                        │
        │  ┌──────────────────────────────────┐ │
        │  │6. INDEX CREATION (POST-PROCESS)  │ │
        │  ├──────────────────────────────────┤ │
        │  │ After all vectors uploaded:      │ │
        │  │  - CREATE INDEX HNSW             │ │
        │  │  - ON embedding VECTOR column    │ │
        │  │  - With cosine distance          │ │
        │  │ Time: 30-60 min (600k vectors)   │ │
        │  │ Result: ~50-100ms query latency  │ │
        │  └──────────────────────────────────┘ │
        │                                        │
        ╚════════════════════════════════════════╝
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│            SUPABASE pgvector (500MB Free Tier)                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Table: document_embeddings                              │   │
│  │  ├─ chunk_id: TEXT (unique)                            │   │
│  │  ├─ text: TEXT (10,000 chars max)                      │   │
│  │  ├─ embedding: VECTOR(768) ← Main column               │   │
│  │  ├─ source_pdf: TEXT (path metadata)                   │   │
│  │  ├─ page_num: INT (page in PDF)                        │   │
│  │  ├─ clasa: INT (class 0-4)                             │   │
│  │  ├─ materie: TEXT (subject)                            │   │
│  │  ├─ capitol: TEXT (chapter)                            │   │
│  │  ├─ chunk_hash: TEXT (MD5, for dedup)                  │   │
│  │  └─ created_at: TIMESTAMP                              │   │
│  │                                                         │   │
│  │ Indexes:                                               │   │
│  │  - HNSW on embedding (cosine similarity)              │   │
│  │  - Standard on clasa, materie, source_pdf             │   │
│  │                                                         │   │
│  │ Functions (RPC):                                       │   │
│  │  - match_documents(query, match_count, filters)       │   │
│  │  - get_statistics()                                    │   │
│  │  - count_vectors()                                     │   │
│  │                                                         │   │
│  │ Stats:                                                 │   │
│  │  - Total vectors: 400k-600k                           │   │
│  │  - Database size: ~300-500 MB                         │   │
│  │  - Vector storage: 768 floats × N × 4 bytes = ~2MB/k  │   │
│  │  - Index size: ~100-150 MB (HNSW overhead)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─── [READY FOR AI APPLICATION]
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│         YOUR AI TUTORING SYSTEM (Separate Repository)           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │1. User query: "Cum se calculează aria unui pătrat?"    │   │
│  │2. Generate embedding cu same model                     │   │
│  │3. Query Supabase: match_documents(query_emb, top=10)   │   │
│  │4. Retrieve top-10 similar chunks + metadata            │   │
│  │5. Format as context pentru LLM (GPT/Claude)            │   │
│  │6. LLM generates tutoring response                      │   │
│  │7. User gets personalized answer!                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Detailed Component Explanation

### 1. **PDF EXTRACTION (scripts/pdf_extractor.py)**

**Tool:** PyMuPDF (fitz library)

**De ce PyMuPDF și nu PyPDF2?**
- 10x mai rapid (~500 PDFs/hour vs 50)
- Nativ suportă imagini (extract, analyze)
- Mai stabil cu PDFs educaționale (diagrame)

**Process:**
```python
# 1. Open PDF
pdf = fitz.open("manual.pdf")

# 2. For each page:
for page in pdf:
    # Extract text
    text = page.get_text()

    # Get images
    images = page.get_images()

    # For each image:
    for img_ref in images:
        pixmap = fitz.Pixmap(pdf, img_ref)
        # Analyze: size, aspect ratio, type
        # Store metadata

# 3. Return
result = {
    "text": "...",  # All text combined
    "images": [
        {
            "page": 1,
            "size": 102400,  # bytes
            "priority_score": 0.9,  # High priority (diagram)
            "type": "diagram"
        },
        ...
    ],
    "total_pages": 45,
    "file_size": 2500000  # bytes
}
```

**Performance:** ~100-200ms per PDF (average 2-10 MB)

---

### 2. **OCR PROCESSING (scripts/ocr_processor.py)**

**Tool:** PaddleOCR (paddle-paddle library)

**De ce PaddleOCR?**
- Gratuit (vs Google Cloud Vision = $3-6 per 1000 images)
- CPU-based (funcționează pe Kaggle CPU zones fără GPU)
- 80+ limbi (inclusiv română)
- Mai bun pentru diagrame educaționale vs Tesseract

**Process:**
```python
# 1. Load PaddleOCR model (first run = download ~500MB)
ocr = PaddleOCR(lang=['ro', 'en'])

# 2. For each image (selective based on priority):
if image.priority_score > 0.8:
    result = ocr.ocr(image_array)

    # result = [
    #     [
    #         ([[x1, y1], [x2, y2], ...], ("text", confidence)),
    #         ...
    #     ]
    # ]

# 3. Extract text + confidence
ocr_text = "\n".join([line[1][0] for line in result[0]])

# 4. Merge with PDF text
combined_text = pdf_text + "\n" + ocr_text
```

**Performance:**
- ~10-30 seconds per image (CPU)
- ~2-5 seconds per image (GPU - but we skip GPU for stability)
- Selective: 60% images skipped (low priority)

---

### 3. **TEXT CHUNKING (scripts/chunking.py)**

**De ce chunking?**
- Embeddings au context window (token limit)
- Matching: full PDF = 10,000+ chars → exceeds token limit
- Solution: split în chunks, each ~500 chars

**Strategy:**
```
Raw text: "Aria pătrat = latura²... (1000 chars total) ...perimetru"

Chunking:
┌─────────────────────────────────┐
│ Chunk 1 (500 chars)              │
│ "Aria pătrat = latura²..."       │
└─────────────────────────────────┘
         │
         ├─ Overlap (50 chars)
         │
         ↓
┌─────────────────────────────────┐
│ Chunk 2 (500 chars)              │
│ "...atura²... perimetru..."      │
└─────────────────────────────────┘

Overlap ensures: semantic continuity between chunks
```

**Deduplication:**
- Calculate MD5 hash: hash("text content")
- Skip if hash seen before
- Eliminates: headers, footers, repeated disclaimers

**Output:** List[Chunk]
```python
[
    Chunk(
        text="Aria pătrat = latura²...",
        chunk_id="chunk_001",
        chunk_hash="abc123...",
        source_page=1,
        metadata={'char_count': 500, ...}
    ),
    ...
]
```

---

### 4. **EMBEDDING GENERATION (scripts/embedding_generator.py)**

**Model:** `paraphrase-multilingual-mpnet-base-v2`

**Architecture:**
```
Text Input (max 384 tokens)
    ↓
Tokenizer (BERT)
    ↓
Embedding Layer (768 hidden units)
    ↓
Transformer Blocks (×12 layers)
    ↓
Mean Pooling (aggregate all tokens)
    ↓
Output: Vector (768 dimensions)
```

**Why 768 dimensions?**
- 384 dimensions = too small (loses semantic nuances)
- 1024+ dimensions = too large (slow, memory intensive)
- 768 = Goldilocks zone for semantic similarity + efficiency

**Performance:**
- Single text: ~50ms (CPU)
- Batch 128: ~200ms total = 1.5ms/text (GPU batching)
- Rate: ~50,000 vectors/hour (GPU)

**Quality:**
- Multi-lingual: handles Romanian + English seamlessly
- Semantic: "aria pătrat" ≈ embedding similar to "square area"
- Robust: handles misspellings, synonyms

---

### 5. **BATCH UPLOAD (scripts/supabase_uploader.py)**

**Why batching?**
- 600k individual inserts = 600k API calls = hours
- 60 batches × 10k vectors = 60 API calls = minutes
- 10x faster + less network overhead

**Upload sequence:**
```
Batch 1: vectors 1-10k      → 20-30 seconds
Batch 2: vectors 10k-20k    → 20-30 seconds
...
Batch 60: vectors 590k-600k → 20-30 seconds
---
Total: ~20 minutes (vectori) + 30-60 min (index creation)
```

**Retry logic:**
```python
for attempt in range(3):  # Max 3 attempts
    try:
        upload_batch()
        break  # Success, move to next batch
    except Exception:
        if attempt < 2:
            sleep(5)  # Wait before retry
        else:
            failed_count += 1  # Record failure
```

---

### 6. **INDEX CREATION (Supabase SQL)**

**Why HNSW Index?**
- Without: similarity query = full table scan = 10+ seconds
- With HNSW: ~50-100ms per query (600k vectors)

**HNSW (Hierarchical Navigable Small World):**
- Graph-based nearest neighbor search
- Trade-off: 15% memory overhead for 100x speed
- Better than: linear scan, LSH, or PQ for embeddings

**SQL:**
```sql
CREATE INDEX idx_embedding_hnsw
ON document_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Parameters:**
- `m = 16`: graph connectivity (higher = more accurate but more memory)
- `ef_construction = 64`: construction parameter (higher = better index)

**Index building time:**
- 100k vectors: ~5 minutes
- 600k vectors: ~30-60 minutes
- One-time cost (done once after upload)

---

## 📊 Data Flow Example

Let's trace one complete document:

```
INPUT PDF: "clasa_1/matematica/cap_01_adunare.pdf" (3.2 MB, 45 pages)

1. PDF EXTRACTION
   ├─ Extract pages 1-45: "1 + 1 = 2. 2 + 3 = 5..."
   ├─ Find 8 images (diagrams with numbers)
   ├─ Filter 3 images (too small or decorative)
   └─ Output: {text: "4500 chars", images: [5 diagrams]}

2. OCR PROCESSING
   ├─ Process 5 images (priority > 0.8)
   ├─ Extract: "Exercise 1: Calculate 5+7"
   ├─ Merge with PDF text
   └─ Output: {text: "5200 chars" (with OCR)}

3. TEXT CHUNKING
   ├─ Split at periods: [sent1, sent2, ..., sent50]
   ├─ Group into chunks of ~500 chars
   ├─ Add overlaps (50 chars)
   ├─ Remove duplicates (footer on all pages)
   └─ Output: [Chunk1, Chunk2, ..., Chunk11] (11 chunks)

4. EMBEDDING GENERATION
   ├─ Batch all chunks: 128 at a time (but only 11 here)
   ├─ Tokenize: split text into tokens (max 384)
   ├─ Forward pass: 11 texts → 11 vectors (768-dim each)
   └─ Output: numpy array (11, 768)

5. METADATA ENRICHMENT
   ├─ chunk_id: "clasa_1_matematica_cap01_chunk_1" ... "chunk_11"
   ├─ source_pdf: "clasa_1/matematica/cap_01_adunare.pdf"
   ├─ page_num: 1, 3, 5, 8, ... (page where chunk from)
   ├─ clasa: 1
   ├─ materie: "Matematică"
   ├─ capitol: "Capitolul 1 - Adunare"
   └─ has_images: true (because used OCR)

6. SUPABASE UPLOAD
   ├─ Format: [
   │   {
   │     "chunk_id": "clasa_1_matematica_cap01_chunk_1",
   │     "text": "1 + 1 = 2. 2 + 2 = 4...",
   │     "embedding": "[0.123, 0.456, ..., -0.789]",  // 768 floats
   │     "source_pdf": "clasa_1/matematica/cap_01_adunare.pdf",
   │     "page_num": 1,
   │     "clasa": 1,
   │     "materie": "Matematică",
   │     "capitol": "Capitolul 1 - Adunare",
   │     "chunk_hash": "a1b2c3d4e5f6...",  // MD5(text)
   │     "has_images": true
   │   },
   │   ...  // 10 more chunks
   │ ]
   │
   ├─ Batch insert: 11 rows inserted into DB
   └─ Status: "OK", 11/11 inserted

RESULT in Supabase:
┌──────────┬────────────────────────────┬──────────────────┬───────┬───────────────┐
│ chunk_id │ text                       │ embedding        │ clasa │ materie       │
├──────────┼────────────────────────────┼──────────────────┼───────┼───────────────┤
│ ..._1    │ "1 + 1 = 2..."             │ [0.123, ...]     │ 1     │ Matematică    │
│ ..._2    │ "2 + 3 = 5..."             │ [0.456, ...]     │ 1     │ Matematică    │
│ ...      │ ...                        │ ...              │ ...   │ ...           │
└──────────┴────────────────────────────┴──────────────────┴───────┴───────────────┘
```

---

## 🎯 Query Time (Usage in AI App)

```python
# User: "Cum se calculează suma 5 + 7?"

# 1. Generate query embedding (same model as training)
query_text = "Cum se calculează suma 5 + 7?"
query_embedding = embedding_model.encode([query_text])[0]  # vector(768)

# 2. Query Supabase
results = supabase.rpc('match_documents', {
    'query_embedding': query_embedding.tolist(),
    'match_count': 10,
    'filter_clasa': 1,  # Only class 1
    'filter_materie': 'Matematică'
})

# 3. Get results
results = [
    {
        'text': '5 + 7 = 12',
        'similarity': 0.92,  # cosine similarity
        'metadata': {
            'source_pdf': 'clasa_1/matematica.pdf',
            'page_num': 15,
            'capitol': 'Adunare'
        }
    },
    {
        'text': 'Exercise: Calculate 5 + 7 + 3',
        'similarity': 0.85,
        ...
    },
    ...  # up to 10 results
]

# 4. Format for LLM
context = "\n".join([f"- {r['text']}" for r in results])
prompt = f"""
Răspunde la întrebarea: "Cum se calculează suma 5 + 7?"
Folosind materialele din manual:
{context}
"""

# 5. Call LLM (GPT-4, Claude, etc)
response = llm.generate(prompt)
# Output: "Suma 5 + 7 = 12. Pentru a calcula:
#         5 + 5 = 10, plus 2 mai = 12."
```

**Latency breakdown:**
- Generate query embedding: ~50ms (CPU)
- Query Supabase: ~100ms (HNSW search)
- Retrieve results: ~50ms (network)
- **Total: ~200ms before LLM**

---

## 🔐 Database Security

**Current (Free Tier):**
- Anon key: public, safe for SELECT
- Service role key: secret, never in frontend
- RLS: Disabled (public read-only access)

**For Production:**
- Enable RLS (Row Level Security)
- Create policies per user/role
- Use service role key only for backend
- Rotate keys periodically

---

## 📈 Scalability

**Current: 600k vectors × 768 dimensions**

```
Storage calculation:
- Vector: 768 floats × 4 bytes = 3.07 KB
- Metadata: ~500 bytes
- Total per row: ~3.5 KB
- 600k rows × 3.5 KB = ~2.1 GB (but Supabase stores compressed ~300-500 MB)

Query performance:
- HNSW index: ~100ms for top-10 from 600k
- Linear scaling: 10M vectors → ~100-200ms (index scales efficiently)

Limits on Supabase free tier:
- 500 MB storage: supports ~150-170k vectors (uncompressed)
- But actual usage: ~300-500 MB (compressed pgvector)
- Current 600k: close to limit
```

**If you need >600k vectors:**
1. Upgrade Supabase to paid plan ($15/month → 100GB)
2. Or split into multiple projects
3. Or move to managed vector database (Pinecone, Weaviate)

---

## 🛠️ Troubleshooting Flow

```
Problem: "Slow queries"
→ Check HNSW index exists?
  ```sql
  SELECT * FROM pg_indexes WHERE tablename = 'document_embeddings'
  ```
→ If no HNSW index: run CREATE INDEX
→ If index exists: check query plan
  ```sql
  EXPLAIN ANALYZE SELECT * FROM match_documents(...)
  ```

Problem: "Out of memory during processing"
→ Reduce batch size: embeddings.batch_size = 32
→ Reduce chunk overlap: pdf.overlap = 20
→ Process PDFs in smaller groups (checkpoint system)

Problem: "Upload fails on specific vectors"
→ Check vector dimensions: should be exactly 768
→ Check text encoding: UTF-8
→ Check for NULL values in required columns
→ Verify API key permissions
```

---

## 📚 Next Steps

Once embeddings are in Supabase:

1. **AI Tutoring App:** Use match_documents() for context retrieval
2. **RAG (Retrieval Augmented Generation):** Feed retrieval results to LLM
3. **Search UI:** Allow users to query embeddings directly
4. **Analytics:** Track which topics are searched most
5. **Recommendations:** Suggest similar topics to users

---

**Status: Embeddings Processing Pipeline Ready!**
