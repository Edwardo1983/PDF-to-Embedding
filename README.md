# 📚 PDF-to-Embedding Pipeline

> **[🇷🇴 Română](#readme-română) | [🇬🇧 English](#readme-english)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20P100-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![Supabase](https://img.shields.io/badge/Supabase-pgvector-3ECF8E?logo=supabase)](https://supabase.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Transform 15GB of educational PDF manuals into 400k-600k semantic vector embeddings for AI-powered tutoring systems.**

Complete pipeline for processing PDFs → Text Extraction → OCR → Chunking → Embeddings → Vector Database (Supabase pgvector), ready for Retrieval-Augmented Generation (RAG) applications.

---

## 🌍 Language / Limbă

<details open>
<summary><b>🇷🇴 Citește în Română (Click to expand)</b></summary>

# README Română

## 📖 Despre Proiect

**PDF-to-Embedding** este un pipeline complet și gratuit pentru transformarea documentelor PDF educaționale în embeddings semantice, gata pentru sisteme AI de tutoriat.

### 🎯 Ce Face?

Transformă **15GB de manuale școlare PDF** (clasele 0-4) în **~400k-600k vectori semantici** stocați permanent în Supabase pgvector, ready pentru:
- ✅ Sisteme RAG (Retrieval-Augmented Generation)
- ✅ Căutare semantică în documente
- ✅ Chatbots educaționali inteligenti
- ✅ Recomandări de conținut personalizate

### 💡 De Ce Acest Proiect?

**Problema:** AI tutors necesită acces rapid la informații din manuale, dar LLM-urile au context limit.
**Soluția:** Convertim tot conținutul în embeddings → căutare semantică ultra-rapidă → feed relevant context to LLM.

**Cost:** **$0** (Kaggle GPU free + Supabase free tier)
**Timp:** ~24 ore procesare (overnight, automat)
**Rezultat:** Vector database permanent, 500MB, query latency ~50-100ms

---

## 🎓 Nivel de Competență Necesar

### Skill Level: **Beginner-Friendly** ⭐⭐☆☆☆

**Nu trebuie să fii expert!** Proiectul e construit pentru **automation engineers** și **începători în Python**.

| Skill | Nivel Necesar | Note |
|-------|---------------|------|
| **Python** | Începător | Copy-paste cod, rulare comenzi simple |
| **Git/GitHub** | Opțional | Doar pentru contribuții (nu e necesar) |
| **SQL** | Zero | SQL-ul e gata scris, doar copy-paste |
| **Machine Learning** | Zero | Modelele pre-trained sunt folosite automat |
| **Cloud Services** | Începător | Ghid pas-cu-pas pentru Kaggle + Supabase |

### 🤖 **Recomandare: Folosește AI Assistants!**

**Acest proiect a fost construit CU și PENTRU AI assistance.**

✅ **Claude Code** (recomandat) - pentru debugging, explicații cod
✅ **ChatGPT** - pentru întrebări generale
✅ **GitHub Copilot** - pentru completări cod (opțional)

**Exemplu workflow cu Claude Code:**
```
Tu: "Am eroarea X la instalare"
Claude Code: [Analizează eroarea, dă soluție pas-cu-pas]

Tu: "Explică ce face funcția extract_text_and_images"
Claude Code: [Explicație detaliată în română + exemple]

Tu: "Cum modific chunk_size să fie 300 în loc de 500?"
Claude Code: [Arată exact ce să editezi în config.yaml]
```

**Dacă te blochezi:** Pune întrebarea unui AI assistant. **E 100% OK!**

---

## 🏗️ Stack Tehnologic

| Componentă | Tool | De Ce? | Alternativă |
|-----------|------|--------|-------------|
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io/) | 10x mai rapid decât PyPDF2 | PyPDF2, pdfplumber |
| **OCR** | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Gratuit, română support, CPU-friendly | Tesseract, Google Vision API |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) | Multilingual, 768-dim, proven accuracy | OpenAI Embeddings ($$$) |
| **Vector DB** | [Supabase pgvector](https://supabase.com/docs/guides/ai) | 500MB permanent free, HNSW indexing | Pinecone, Weaviate, Qdrant |
| **GPU** | [Kaggle P100](https://www.kaggle.com/) | 30h/săptămână gratuit | Google Colab (12h/zi) |
| **Notebook** | Jupyter | Executare vizuală pas-cu-pas | Python scripts |

---

## ⏱️ Timeline Estimat

| Etapă | Durată | Când | Automatizat? |
|-------|--------|------|--------------|
| **Setup conturi** | 15 min | Acum | ❌ Manual |
| **Upload PDFs** | 1-2 ore | Overnight | ✅ Da |
| **Procesare Kaggle** | 18-24 ore | Overnight | ✅ Da |
| **Create Index** | 30-60 min | După procesare | ✅ Da |
| **TOTAL** | ~26 ore | 2-3 zile | 95% automat |

💡 **Timp efectiv petrecut de tine:** ~30 minute (setup + monitoring)

---

## 💰 Cost Breakdown

```
┌─────────────────────────────────────────────┐
│ TOTAL COST: $0                              │
├─────────────────────────────────────────────┤
│ Kaggle P100 GPU (30h/week):        $0      │
│ Supabase pgvector (500MB):         $0      │
│ sentence-transformers model:       $0      │
│ PaddleOCR:                         $0      │
│ Bandwidth (upload 15GB):           $0      │
└─────────────────────────────────────────────┘

⚡ Comparație:
- OpenAI Embeddings: ~$60-80 pentru 600k chunks
- Google Vision OCR: ~$15-30 pentru imagini
- Pinecone (vector DB): ~$70/month
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SALVEZI: ~$150+ cu acest pipeline gratuit
```

---

## 📋 Prerequisites

### 1️⃣ **Conturi Online (Gratuite)**

| Serviciu | Link Signup | Timp Setup | Phone Needed? |
|----------|-------------|------------|---------------|
| **Kaggle** | [kaggle.com/account/login](https://www.kaggle.com/account/login) | 5 min | ✅ Da (pentru GPU) |
| **Supabase** | [supabase.com](https://supabase.com) | 5 min | ❌ Nu |
| **GitHub** | [github.com/signup](https://github.com/signup) | 5 min (opțional) | ❌ Nu |

### 2️⃣ **Software Local**

**Windows 10/11:**
```powershell
# Verifică Python
python --version
# Expected: Python 3.9.x sau 3.10.x sau 3.11.x

# Dacă nu ai Python:
# 1. Download: https://www.python.org/downloads/
# 2. Instalează cu "Add Python to PATH" ✅ bifat
# 3. Restart PowerShell
```

**macOS:**
```bash
# Verifică Python
python3 --version

# Instalare dacă lipsește:
brew install python@3.10
```

**Linux (Ubuntu/Debian):**
```bash
# Verifică Python
python3 --version

# Instalare dacă lipsește:
sudo apt update
sudo apt install python3.10 python3-pip
```

### 3️⃣ **Materiale PDF (Opțional pentru test)**

- Pentru **test local:** 2-3 PDFs mici (1-5 MB fiecare)
- Pentru **procesare completă:** 15GB PDFs organizate în `materiale_didactice/`

**Structură recomandată:**
```
materiale_didactice/
├── clasa_0/
│   ├── matematica/
│   │   └── capitol_1.pdf
│   └── romana/
├── clasa_1/
...
```

---

## 🚀 Quick Start (3 Pași Simpli)

### **Pas 1: Clone Repository** (2 min)

```bash
# Clone proiect
git clone https://github.com/Edwardo1983/PDF-to-Embedding.git
cd PDF-to-Embedding

# Instalează dependencies (poate lua 5-10 min)
pip install -r config/requirements-minimal.txt
```

### **Pas 2: Setup Supabase** (10 min)

1. **Creează cont:** [supabase.com/dashboard](https://supabase.com/dashboard)
2. **New Project** → Alege region EU → Așteaptă 2 min
3. **SQL Editor** → Copy-paste tot din `sql/supabase_setup.sql` → Run
4. **Settings → API** → Copy:
   - `Project URL`
   - `anon public key`

### **Pas 3: Upload & Process pe Kaggle** (30 min setup + 24h procesare)

1. **Upload PDFs:**
   - [kaggle.com/datasets/create](https://www.kaggle.com/datasets/create)
   - Upload folder `materiale_didactice/`
   - Set **Private**

2. **Create Notebook:**
   - [kaggle.com/code/create](https://www.kaggle.com/code/create)
   - Settings → **GPU (P100)** ✅
   - Settings → Secrets → Add `SUPABASE_URL` și `SUPABASE_ANON_KEY`

3. **Copy-Paste Cod:**
   - Deschide `kaggle_notebook.ipynb`
   - Copy tot codul în Kaggle
   - Click **"Run All"**

4. **Lasă să proceseze overnight** (~24h)

✅ **Gata!** Embeddings-urile tale sunt în Supabase, permanent.

---

## 📚 Documentație Completă

| Document | Descriere | Când să-l citești |
|----------|-----------|-------------------|
| **[SETUP_GUIDE.md](SETUP_GUIDE.md)** | Pas-cu-pas detaliat Kaggle + Supabase | La început (mandatory) |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Flow tehnic, diagrame, explicații | Pentru înțelegere deep |
| **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | 20+ soluții probleme comune | Când întâmpini erori |

---

## 🧪 Test Local (Opțional, 30 min)

**Înainte de procesare completă, testează cu 2-3 PDFs:**

```bash
# 1. Adaugă PDFs de test
copy your-test.pdf materiale_didactice/

# 2. Configurează Supabase (dacă vrei să testezi upload)
notepad .env
# Adaugă:
# SUPABASE_URL=https://xxxxx.supabase.co
# SUPABASE_ANON_KEY=eyJhbGci...

# 3. Rulează test
python tests/test_sample.py
```

**Expected output:**
```
✅ TEST 1: PDF EXTRACTION - PASSED
✅ TEST 2: OCR PROCESSING - PASSED (skipped, OCR disabled)
✅ TEST 3: TEXT CHUNKING - PASSED
✅ TEST 4: EMBEDDING GENERATION - PASSED
✅ TEST 5: SUPABASE CONNECTION - PASSED
✅ ALL TESTS PASSED!
```

---

## 🎯 Output Final

După procesare, vei avea:

```
📊 Supabase Database:
├─ ~400k-600k vector embeddings
├─ 768 dimensions (multilingual semantic)
├─ HNSW index (cosine similarity)
├─ Query latency: 50-100ms
├─ Storage: ~300-500 MB
└─ Status: ✅ Permanent (free tier)

🔍 Capabilities:
├─ Semantic search: "Cum se calculează aria?"
├─ Similarity matching: top-K results
├─ Metadata filtering: by class, subject, chapter
└─ Ready for RAG integration with LLMs
```

---

## 🛠️ Cum Folosești Embeddings-urile?

**Exemplu Python (în aplicația ta de tutoriat):**

```python
from supabase import create_client
from sentence_transformers import SentenceTransformer

# 1. Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Load embedding model (same as pipeline)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 3. User question
user_query = "Cum se calculează aria unui pătrat?"

# 4. Generate query embedding
query_embedding = model.encode([user_query])[0].tolist()

# 5. Search similar chunks in Supabase
results = supabase.rpc('match_documents', {
    'query_embedding': query_embedding,
    'match_count': 10,
    'filter_clasa': 1,  # Optional: filter by class
    'filter_materie': 'Matematică'  # Optional: filter by subject
}).execute()

# 6. Use top results as context for LLM
context = "\n".join([r['text'] for r in results.data[:5]])

# 7. Send to LLM (GPT-4, Claude, etc)
llm_response = your_llm_function(
    system="You are an educational tutor.",
    user_query=user_query,
    context=context
)

print(llm_response)
# Output: "Aria pătratului se calculează înmulțind latura cu ea însăși:
#          Aria = latura × latura sau latura².
#          De exemplu, dacă latura = 5cm, atunci Aria = 25cm²"
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────┐
│   15GB PDFs LOCAL   │
│  (materiale_did.)   │
└──────────┬──────────┘
           │
           ├─ [UPLOAD]
           ↓
┌─────────────────────┐
│   KAGGLE DATASET    │
│  (mounted as input) │
└──────────┬──────────┘
           │
           ├─ [PROCESSING PIPELINE]
           ↓
    ┌──────────────┐
    │  PyMuPDF     │  → Extract text + images
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  PaddleOCR   │  → OCR text from diagrams
    └──────┬───────┘
           │
    ┌──────▼────────┐
    │   Chunking    │  → Split into 500-char chunks
    └──────┬────────┘
           │
    ┌──────▼─────────────┐
    │ sentence-transform │  → Generate 768-dim embeddings
    └──────┬─────────────┘
           │
           ├─ [BATCH UPLOAD 10k]
           ↓
┌──────────────────────────┐
│  SUPABASE pgvector       │
│  ┌────────────────────┐  │
│  │ 600k vectors       │  │
│  │ HNSW index         │  │
│  │ RPC functions      │  │
│  └────────────────────┘  │
└──────────────────────────┘
           │
           ├─ [QUERY API]
           ↓
┌──────────────────────────┐
│   YOUR AI TUTOR APP      │
│   (RAG System)           │
└──────────────────────────┘
```

---

## 🤝 Contributing

Contribuțiile sunt binevenite! Dacă îmbunătățești ceva:

1. Fork repository
2. Create branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

**Ideas for contributions:**
- ✅ Support pentru alte limbi (spaniolă, franceză, etc)
- ✅ Improved OCR quality detection
- ✅ Web UI pentru monitoring procesare
- ✅ Docker containerization
- ✅ Traduceri README în alte limbi

---

## 🐛 Troubleshooting

**Problem:** "GPU not available" în Kaggle
**Solution:** [TROUBLESHOOTING.md#gpu-not-available](docs/TROUBLESHOOTING.md#gpu-not-available)

**Problem:** "Out of memory"
**Solution:** [TROUBLESHOOTING.md#out-of-memory](docs/TROUBLESHOOTING.md#out-of-memory)

**Problem:** "Supabase connection timeout"
**Solution:** [TROUBLESHOOTING.md#supabase-connection](docs/TROUBLESHOOTING.md#supabase-connection)

📖 **[View All Solutions →](docs/TROUBLESHOOTING.md)**

---

## 📞 Support & Community

- 💬 **GitHub Issues:** [Report bugs](https://github.com/Edwardo1983/PDF-to-Embedding/issues)
- 📧 **Email:** (Add your email dacă vrei)
- 🤖 **AI Assistance:** Use Claude Code, ChatGPT pentru debugging

---

## 📜 License

MIT License - Free to use, modify, distribute.

See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

**Built with:**
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF processing
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings
- [Supabase](https://supabase.com/) - Vector database
- [Kaggle](https://www.kaggle.com/) - Free GPU compute

**Created by:** Edd - Automation Engineer
**AI Assisted:** Claude Code (Anthropic)

---

## ⭐ Star History

If this project helped you, consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=Edwardo1983/PDF-to-Embedding&type=Date)](https://star-history.com/#Edwardo1983/PDF-to-Embedding&Date)

---

</details>

---

<details>
<summary><b>🇬🇧 Read in English (Click to expand)</b></summary>

# README English

## 📖 About

**PDF-to-Embedding** is a complete, free pipeline for transforming educational PDF documents into semantic embeddings, ready for AI tutoring systems.

### 🎯 What It Does

Transforms **15GB of educational PDF manuals** (grades 0-4) into **~400k-600k semantic vectors** permanently stored in Supabase pgvector, ready for:
- ✅ RAG (Retrieval-Augmented Generation) systems
- ✅ Semantic document search
- ✅ Intelligent educational chatbots
- ✅ Personalized content recommendations

### 💡 Why This Project?

**Problem:** AI tutors need quick access to manual information, but LLMs have context limits.
**Solution:** Convert all content into embeddings → ultra-fast semantic search → feed relevant context to LLM.

**Cost:** **$0** (Kaggle free GPU + Supabase free tier)
**Time:** ~24 hours processing (overnight, automated)
**Result:** Permanent vector database, 500MB, query latency ~50-100ms

---

## 🎓 Required Skill Level

### Skill Level: **Beginner-Friendly** ⭐⭐☆☆☆

**You don't need to be an expert!** This project is built for **automation engineers** and **Python beginners**.

| Skill | Required Level | Notes |
|-------|---------------|-------|
| **Python** | Beginner | Copy-paste code, run simple commands |
| **Git/GitHub** | Optional | Only for contributions (not required) |
| **SQL** | Zero | SQL is pre-written, just copy-paste |
| **Machine Learning** | Zero | Pre-trained models used automatically |
| **Cloud Services** | Beginner | Step-by-step guide for Kaggle + Supabase |

### 🤖 **Recommendation: Use AI Assistants!**

**This project was built WITH and FOR AI assistance.**

✅ **Claude Code** (recommended) - for debugging, code explanations
✅ **ChatGPT** - for general questions
✅ **GitHub Copilot** - for code completions (optional)

**Example workflow with Claude Code:**
```
You: "I'm getting error X during installation"
Claude Code: [Analyzes error, provides step-by-step solution]

You: "Explain what the extract_text_and_images function does"
Claude Code: [Detailed explanation with examples]

You: "How do I change chunk_size to 300 instead of 500?"
Claude Code: [Shows exactly what to edit in config.yaml]
```

**If you get stuck:** Ask an AI assistant. **It's 100% OK!**

---

## 🏗️ Tech Stack

| Component | Tool | Why? | Alternative |
|-----------|------|------|-------------|
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io/) | 10x faster than PyPDF2 | PyPDF2, pdfplumber |
| **OCR** | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Free, multilingual, CPU-friendly | Tesseract, Google Vision API |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) | Multilingual, 768-dim, proven accuracy | OpenAI Embeddings ($$$) |
| **Vector DB** | [Supabase pgvector](https://supabase.com/docs/guides/ai) | 500MB permanent free, HNSW indexing | Pinecone, Weaviate, Qdrant |
| **GPU** | [Kaggle P100](https://www.kaggle.com/) | 30h/week free | Google Colab (12h/day) |
| **Notebook** | Jupyter | Visual step-by-step execution | Python scripts |

---

## 🚀 Quick Start

### **Step 1: Clone Repository** (2 min)

```bash
git clone https://github.com/Edwardo1983/PDF-to-Embedding.git
cd PDF-to-Embedding
pip install -r config/requirements-minimal.txt
```

### **Step 2: Setup Supabase** (10 min)

1. Create account: [supabase.com](https://supabase.com)
2. New Project → EU region → Wait 2 min
3. SQL Editor → Copy-paste from `sql/supabase_setup.sql` → Run
4. Settings → API → Copy URL + anon key

### **Step 3: Process on Kaggle** (30 min setup + 24h processing)

1. Upload PDFs to [Kaggle Datasets](https://www.kaggle.com/datasets/create)
2. Create [Kaggle Notebook](https://www.kaggle.com/code/create)
3. Enable **GPU (P100)**
4. Add Secrets: `SUPABASE_URL`, `SUPABASE_ANON_KEY`
5. Copy-paste code from `kaggle_notebook.ipynb`
6. Click **"Run All"**
7. Let it run overnight (~24h)

✅ **Done!** Your embeddings are in Supabase, permanently.

---

## 📚 Full Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed step-by-step setup
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical flow, diagrams
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - 20+ solutions to common issues

---

## 📊 Expected Output

After processing:

- **~400k-600k vector embeddings** (768 dimensions)
- **HNSW index** for fast similarity search
- **Query latency:** 50-100ms
- **Storage:** ~300-500 MB (permanent free tier)
- **Ready for RAG** with any LLM

---

## 🤝 Contributing

Contributions welcome! See issues or open a PR.

---

## 📜 License

MIT License - Free to use, modify, distribute.

---

## 🙏 Credits

Built with PyMuPDF, PaddleOCR, sentence-transformers, Supabase, Kaggle.
**Created by:** Edd - Automation Engineer
**AI Assisted:** Claude Code (Anthropic)

---

</details>

---

## 🌟 Support This Project

If this helped you, please ⭐ **star this repository**!

**Questions?** Open an [issue](https://github.com/Edwardo1983/PDF-to-Embedding/issues) or ask Claude Code!

---

**Last Updated:** November 2024
**Status:** ✅ Production Ready
**Maintained:** Yes
