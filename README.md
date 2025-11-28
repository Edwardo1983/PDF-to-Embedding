# 📚 PDF-to-Embedding: Conversie Manuale Școlare → AI Embeddings

Proiect complet pentru procesare a 15GB manuale didactice PDF (clasele 0-4) și convertire în vector embeddings pentru sistem de tutoriat AI educațional.

---

## 🎯 Overview

### Ce face proiectul?

**Transformă 15GB de PDF-uri şcolare în ~400k-600k vector embeddings gata pentru căutare semantică în AI.**

- 📄 **Input:** 15GB PDF-uri (60% text + 40% imagini educaționale)
- ⚙️ **Procesare:** Extract text, OCR imagini, chunking inteligent, generare embeddings
- 💾 **Output:** Vectori permanenți în Supabase pgvector (500MB free tier)
- 🗑️ **Result:** Recover 15GB local după upload

### Stack Tehnologic

| Componență | Tool | De ce? |
|---|---|---|
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io/) | 10x mai rapid decât PyPDF2, extract text + imagini |
| **OCR pentru Imagini** | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Gratuit, suportă română, funcționează pe CPU |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) | Suportă multilingual, model 768-dimensional |
| **Vector DB** | [Supabase pgvector](https://supabase.com/docs/guides/ai/vecs-python-client) | 500MB permanent free, HNSW indexing |
| **GPU Gratuit** | [Kaggle P100](https://www.kaggle.com/settings/account) | 30h/săptămână, suficient pentru 15GB |
| **Notebook** | Jupyter | Executare pas-cu-pas cu progress tracking |

### ⏱️ Estimări

| Task | Durată | Note |
|---|---|---|
| Setup Kaggle + Supabase | 15 min | One-time, copy-paste |
| Upload PDFs (15GB) | 1-2 h | Depinde de conexiune internet |
| PDF Extraction + OCR | 12-18 h | On Kaggle P100 GPU |
| Embedding Generation | 4-6 h | Batch processing 128 chunks/step |
| Upload Supabase + Index | 2-3 h | Batch 10k vectors, HNSW creation |
| **TOTAL** | **20-25 h** | Rulează overnight, ~$0 |

### 💰 Cost Final

```
Kaggle P100 GPU:        $0  (30h/săpt gratuit)
Supabase pgvector:      $0  (500MB permanent free)
Bandwidth:              $0  (dacă upload local PDFs)
---
TOTAL:                  $0-3 (eventual VPN dacă necesar)
```

---

## 📋 Prerequisites

Înainte de a începe, trebuie să ai:

### 1. **Conturi Online**
- ✅ Cont Kaggle (signup gratuit: [kaggle.com](https://www.kaggle.com/settings/account))
- ✅ Cont Supabase (signup gratuit: [supabase.com](https://supabase.com))
- ✅ Phone verification pe Kaggle (pentru acces GPU)

### 2. **Materiale Locale**
- ✅ Folder `materiale_didactice/` cu 15GB PDF-uri organizate
- ✅ Structură recomandată: `materiale_didactice/clasa_X/materie/chapters/`
- ✅ PDFs de test (2-3 fișiere mici pentru testing)

### 3. **Softare Local**
```bash
# Windows (PowerShell sau Command Prompt)
python --version  # Min Python 3.9

# Dacă nu ai Python, download de la python.org
# La instalare, selectează: "Add Python to PATH"
```

### 4. **Conexiune Internet**
- 📶 Min 10 Mbps pentru upload PDFs
- 📡 Stabil (sessionul Kaggle timeout după inactivitate)

---

## 🚀 Quick Start (3 Pași)

### **Pas 1: Preluare Secrets Supabase** (5 min)
```bash
1. Merge la: https://supabase.com/dashboard
2. Click pe project-ul tău
3. Settings → API Keys
4. Copy "Project URL" + "anon public" key
5. Ține-le în clipboard pentru pasul 4
```

### **Pas 2: Upload PDFs în Kaggle** (30-60 min)
```bash
1. Merge la: https://www.kaggle.com/datasets/create/new
2. Click "Add data from your computer"
3. Upload materiale_didactice/ (poate lua timp pentru 15GB)
4. Set "Visibility" = "Private"
5. Note dataset ID-ul (format: username/dataset-name)
```

### **Pas 3: Run Notebook Kaggle** (18-24 h)
```bash
1. Merge la: https://www.kaggle.com/code/create
2. Copy-paste codul din kaggle_notebook.ipynb
3. Add Secrets: SUPABASE_URL, SUPABASE_KEY
4. Click "Run All" → lasă să proceseze overnight
5. Verifică Supabase dashboard pentru vectors
```

**Gata!** Embeddings-urile tale sunt în Supabase, ready pentru AI tutoring system.

---

## 📖 Detailed Setup

**⚠️ Instrucțiuni pas-cu-pas detaliate aici:** [SETUP_GUIDE.md](SETUP_GUIDE.md)

Conține:
- ✅ Screenshot-style walkthroughs (text format)
- ✅ Troubleshooting pentru common issues
- ✅ Verificări de configurare
- ✅ Testing la fiecare pas

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    15GB PDF LOCAL                            │
│           (materiale_didactice/clasa_*/...)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─[MANUAL UPLOAD]
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  KAGGLE DATASET                              │
│     (Your PDFs mounted as /kaggle/input/dataset/)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─[KAGGLE NOTEBOOK PROCESSING]
                     ↓
┌────────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE                         │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │ PyMuPDF      │──→   │ Text Extract │                   │
│  │ (text/img)   │      │ (1000s)      │                   │
│  └──────────────┘      └──────┬───────┘                   │
│                                │                            │
│                         ┌──────▼───────┐                   │
│                         │ PaddleOCR    │                   │
│                         │ (imagini)    │                   │
│                         └──────┬───────┘                   │
│                                │                            │
│                         ┌──────▼────────────┐              │
│                         │ Chunking + Dedup  │              │
│                         │ (500 chars + hash)│              │
│                         └──────┬────────────┘              │
│                                │                            │
│                    ┌───────────▼───────────┐              │
│                    │ sentence-transformers │              │
│                    │ (768-dim vectors)     │              │
│                    └───────────┬───────────┘              │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ├─[BATCH UPLOAD 10k]
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                SUPABASE pgvector (500MB)                     │
│     ┌──────────────────────────────────────────────────┐   │
│     │ 400k-600k VECTORS (768 dim each)                 │   │
│     │ + Metadata: source_pdf, page, clasa, materie    │   │
│     │ + HNSW Index (cosine similarity)                │   │
│     │ + Match RPC function (similarity search)         │   │
│     └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                    ┌────▼─────┐
                    │ Your AI   │
                    │ Tutoring  │
                    │ System    │
                    └──────────┘
```

**Flow Detaliat:**
1. **Extract Text** (PyMuPDF): Parse PDF → text + metapagină
2. **OCR Imagini** (PaddleOCR): Recunoștere text din diagrame
3. **Chunking**: Split text în 500-char chunks cu 50-char overlap
4. **Deduplication**: Hash MD5 pentru skip headers/footers
5. **Embeddings**: Batch processing cu sentence-transformers
6. **Upload Supabase**: 10k vectors per batch cu retry logic
7. **Indexing**: HNSW creation pentru fast similarity search

---

## 📁 Project Structure

```
pdf-to-embedding/
├── README.md                          # ← Ești aici
├── SETUP_GUIDE.md                     # Instrucțiuni detaliate setup
├── kaggle_notebook.ipynb              # Notebook principal (copy-paste în Kaggle)
│
├── scripts/                           # Python modules
│   ├── pdf_extractor.py              # Extract text + imagini din PDFs
│   ├── ocr_processor.py              # PaddleOCR pentru imagini
│   ├── chunking.py                   # Split text inteligent + dedup
│   ├── embedding_generator.py        # sentence-transformers batched
│   └── supabase_uploader.py          # Batch upload cu retry
│
├── config/
│   ├── requirements.txt               # Dependencies (pip install)
│   └── config.yaml                   # Parametri: chunk_size, batch_size, etc
│
├── sql/
│   └── supabase_setup.sql            # Schema pgvector + indexes + RPC
│
├── tests/
│   └── test_sample.py                # E2E test cu 2-3 PDFs mici
│
├── docs/
│   ├── TROUBLESHOOTING.md            # Solutions pentru common issues
│   └── ARCHITECTURE.md               # Explicație detaliată flow
│
└── materiale_didactice/              # ← Upload tine PDFs AICI
    ├── clasa_0/
    ├── clasa_1/
    ├── ...
    └── clasa_4/
```

---

## 🔍 Verificare Setup

După ce completezi SETUP_GUIDE.md, rulează:

```bash
# 1. Check Python version
python --version

# 2. Install dependencies
pip install -r config/requirements.txt

# 3. Run sample test (process 2-3 PDFs mici)
python tests/test_sample.py

# 4. Expected output:
# ✅ Processed 3 PDFs
# ✅ Generated 1,250 embeddings
# ✅ Uploaded to Supabase: 1,250/1,250
# ✅ Sample query test passed
```

---

## 🐛 Troubleshooting

**Problem:** "GPU not available" în Kaggle
**Solution:** [SETUP_GUIDE.md → Pas 1.3 Phone verification](SETUP_GUIDE.md#pas-13-phone-verification)

**Problem:** "Out of memory" la processing
**Solution:** [TROUBLESHOOTING.md → Kaggle Memory Issues](docs/TROUBLESHOOTING.md#out-of-memory-la-processing)

**Problem:** "Supabase connection timeout"
**Solution:** [TROUBLESHOOTING.md → Supabase Connection](docs/TROUBLESHOOTING.md#connection-timeout-la-supabase)

**⚠️ Mai multe soluții:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 📊 Estimări de Output

După procesare completă (~24 ore):

```
INPUT:
- 15GB PDFs
- ~3000-5000 fișiere
- ~800k-1M pagini

PROCESSING STATS:
- Text extraction rate: ~500 PDFs/ora (GPU accelerated)
- OCR images: ~1000 imagini/ora
- Embeddings generation: ~50k vectors/ora
- Upload to Supabase: ~10k vectors/min (batched)

OUTPUT:
- Total chunks: 400k-600k
- Vector dimensions: 768 (multilingual)
- Database size: ~300-500 MB (compressed vectors)
- Storage remaining: ~200MB free Supabase tier
- Query latency: ~50-100ms (HNSW index)

✅ READY FOR: AI tutoring system, semantic search, similarity recommendations
```

---

## 📝 Cum să folosești embeddings-urile

Odată ce embeddings-urile sunt în Supabase, aplicația ta de tutoriat AI poate face:

```python
from supabase import create_client

# 1. Initialize client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Generate query embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
query_embedding = model.encode("Cum se calculează aria unui pătrat?")

# 3. Find similar chunks
results = supabase.rpc('match_documents', {
    'query_embedding': query_embedding.tolist(),
    'match_count': 10,
    'filter_clasa': 1,  # Clasa 1
    'filter_materie': 'Matematică'
}).execute()

# 4. Use results for AI context
for result in results.data:
    print(f"Text: {result['text']}")
    print(f"Similarity: {result['similarity']:.2%}")
    print(f"Source: {result['metadata']['source_pdf']}")
```

---

## 🎓 Educational Use

Acest proiect demonstrează:
- ✅ Pipeline-uri ETL cu Python
- ✅ Processing PDF-uri la scară (15GB+)
- ✅ OCR și NLP cu librării open-source
- ✅ Vector databases și similarity search
- ✅ Kaggle compute resources
- ✅ Batch processing și optimization

Perfect pentru portfolio DevOps/ML Engineering!

---

## 📞 Support

**Probleme tehnice?** Verifică:
1. [SETUP_GUIDE.md](SETUP_GUIDE.md) - Pas-cu-pas complet
2. [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - 20+ soluții
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Flow detaliat

**Vrei să contribui?**
Fork repository-ul și submit pull requests!

---

## 📜 License

MIT - Free to use, modify, distribute

---

## 🚀 Next Steps

✅ **Următor:** [SETUP_GUIDE.md](SETUP_GUIDE.md)
Then: Configure Kaggle + Supabase
Then: Copy-paste notebook
Then: Run overnight
Then: ✨ Embeddings ready!

---

**Creat pentru:** Edd - Automation Engineer
**Scop:** AI Educational Tutoring System (15GB manuale 0-4)
**Status:** Ready for production use

---
