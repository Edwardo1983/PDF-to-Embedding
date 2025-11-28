# 🔧 SETUP_GUIDE.md - Instrucțiuni Pas-cu-Pas

Ghid complet pentru configurarea Kaggle, Supabase și rularea proiectului.

---

## ✅ Checklist Setup Complet

- [ ] **Pas 1:** Kaggle GPU Setup (15 min)
- [ ] **Pas 2:** Upload PDFs în Kaggle Dataset (30-60 min)
- [ ] **Pas 3:** Supabase pgvector Setup (10 min)
- [ ] **Pas 4:** Configurare Notebook Kaggle (5 min)
- [ ] **Pas 5:** Run teste sample (30 min)

**Total timp setup:** ~90-120 min

---

# **PAP 1: KAGGLE GPU SETUP** (15 min)

Kaggle oferă **P100 GPU gratuit** (30h/săptămână). Trebuie verificare telefon.

### **1.1: Creează cont Kaggle**

1. Mergi la [kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Click "Sign up"
3. Completează:
   - Email valid (preferabil Gmail)
   - Parolă puternică (min 8 char, mix numere/simboluri)
   - Display name (ex: "Edd")
4. Click "Sign up"
5. Verifică email (click link din email Kaggle)

```
Expected screen: "Welcome to Kaggle!"
Your dashboard looks like: https://www.kaggle.com/settings/account
```

### **1.2: Phone Verification (Necesar pentru GPU!)**

⚠️ **IMPORTANT:** Fără asta, nu ai acces la P100!

1. Mergi la [kaggle.com/settings/phone](https://www.kaggle.com/settings/phone)
2. Click "Add phone number"
3. Selectează "Romania" (sau țara ta)
4. Introdu numărul tău (format: +40...)
5. Kaggle trimite SMS cu cod
6. Introdu codul
7. Click "Verify"

```
Expected message: ✅ "Phone verified successfully"
Status: Enabled for using accelerators (GPU, TPU)
```

### **1.3: Verificare GPU Availability**

1. Mergi la [kaggle.com/code/create](https://www.kaggle.com/code/create)
2. Click "Create new notebook"
3. În notebook, click **Settings** (icon roată dreapta-sus)

```
BEFORE (cu GPU disabled):
┌─────────────────────┐
│ Accelerator         │
│ [⭕ None selected]  │
└─────────────────────┘

AFTER (cu GPU enabled):
┌─────────────────────┐
│ Accelerator         │
│ [✅ GPU (P100)]     │
│ [Time limit: 9h]    │
└─────────────────────┘
```

4. Click dropdown "None" → selectează "GPU"
5. Confirmă și click "Save"

```
Expected result:
- Accelerator: GPU (NVIDIA Tesla P100)
- Session duration: 9 hours max
- Weekly quota: 30 hours
```

### **1.4: Test GPU în Notebook**

1. Copiază codul în nou notebook:

```python
!nvidia-smi
```

2. Click "Run cell"
3. Verifică output - trebuie să vezi:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.xx.xx    Driver Version: 470.xx.xx                         |
|-------------------------------+----------------------+--------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|  0   Tesla P100-PCIE        Off  | 00000000:00:04.0 Off |                  0 |
| 3B%   35C    P0    75W / 250W |   1050MiB / 16280MiB |      0%      Default |
+-------------------------------+----------------------+--------------------+
```

✅ **SUCCES!** GPU este disponibil.
❌ **ERROR?** Revino la 1.2 - verify phone!

---

# **PAP 2: UPLOAD PDFs ÎN KAGGLE DATASET** (30-60 min)

Trebuie să uploadezi folder-ul `materiale_didactice/` cu toate PDFs.

### **2.1: Creează Kaggle Dataset**

1. Mergi la [kaggle.com/datasets/create/new](https://www.kaggle.com/datasets/create/new)

```
Expected page:
┌───────────────────────────────┐
│ Create a new dataset          │
│                               │
│ Title: [_________________]    │
│ Slug:  [_________________]    │
│ Description: [____________]   │
│                               │
│ Visibility:                   │
│ ⭕ Private ⭕ Public          │
└───────────────────────────────┘
```

2. Completează:
   - **Title:** `Materiale Didactice 0-4`
   - **Slug:** `materiale-didactice-0-4` (auto-generat)
   - **Description:** `Manuale școlare digitale clasele 0-4 pentru AI tutoring`
   - **Visibility:** **Private** (datele educaționale sunt sensibile)

3. Click "Create"

```
Expected result:
Dataset URL: https://www.kaggle.com/datasets/{username}/materiale-didactice-0-4
Status: ✅ Created
```

### **2.2: Upload PDFs**

1. În dataset page, click "Add data from your computer"

```
┌──────────────────────────┐
│ Add data from computer   │
│ [Choose files or dirs]   │ ← Click here
└──────────────────────────┘
```

2. **Selectează folder:** Deschide file explorer
   - Navighează la `C:\Users\Opaop\Desktop\PDF-to-Embedding\materiale_didactice\`
   - Selectează **TOTAL FOLDER** (nu individual PDFs)
   - Ctrl+A pentru tot

3. **Upload starts** - peut durează **30 min - 2 ore** pentru 15GB

```
Upload status:
[████████████░░░░░░░░░░░░░░░░░░] 35% - 5.2 GB / 15 GB
Time remaining: ~45 minutes
```

⚠️ **NU CLOSE TAB!** Upload va continua în background, dar e sigur să lași tab-ul deschis.

### **2.3: Verify Upload Complete**

Când termini upload, ar trebui să vezi:

```
✅ Upload successful
Total files: 3,245
Total size: 15.0 GB

Files structure:
- clasa_0/
  - matematica/
    - cap_01_numere.pdf
    - cap_02_adunare.pdf
  - romana/
    - (...)
- clasa_1/
  - (...)
```

**Dataset ID (rememba asta!):** `{username}/materiale-didactice-0-4`

### **2.4: (Optional) Split în Multiple Datasets dacă upload fails**

Dacă upload-ul de 15GB fail, poti split-a:

```
Dataset 1: clasa_0/ + clasa_1/ = 3.5GB
Dataset 2: clasa_2/ + clasa_3/ = 5.8GB
Dataset 3: clasa_4/ + extras = 5.7GB
```

Apoi în notebook, combini:
```python
# Mount multiple datasets
!kaggle datasets download -d {username}/materiale-didactice-0-4-1
!kaggle datasets download -d {username}/materiale-didactice-0-4-2
```

---

# **PAP 3: SUPABASE PGVECTOR SETUP** (10 min)

Supabase oferă **500MB free tier permanent** cu pgvector support.

### **3.1: Creează cont Supabase**

1. Mergi la [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Alege "Sign up with Email"
4. Completează:
   - Email valid
   - Parolă
5. Verifică email (click link din Supabase)

```
Expected: Supabase dashboard empty
```

### **3.2: Create New Project**

1. Dashboard → Click "New project"

```
┌────────────────────────────┐
│ Create a new project       │
│                            │
│ Name: [______________]     │
│ Database Password:         │
│ [______________] (copy!)   │
│ Region: [dropdown]         │
│                            │
│ [Create project]           │
└────────────────────────────┘
```

2. Completează:
   - **Name:** `pdf-embeddings`
   - **Database Password:** Generează ceva fort (min 12 char) - **ȚINE ASTA SIGUR!**
   - **Region:** Selectează `Europe (EU-West)` sau `Central Europe` (mai apropiat de Romania)

3. Click "Create project"

⏳ **Așteaptă 1-2 minute** - Supabase inițializează database-ul

```
Expected notification:
✅ Project created successfully!
Status: Running
```

### **3.3: Enable pgvector Extension**

1. Mergi la "SQL Editor" (stânga: SQL →)
2. Click "New query"
3. Copiază:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

4. Click "Run"
5. Verifică: `✅ Success. No rows returned`

```
Console output:
CREATE EXTENSION

✅ Extension created
```

### **3.4: Copy API Keys**

1. Mergi la "Settings" (stânga jos)
2. Click "API" submenu

```
┌────────────────────────────────────┐
│ Project API Keys                   │
│                                    │
│ Project URL:                       │
│ https://xxxxx.supabase.co         │ ← COPY THIS
│                                    │
│ Project API Keys:                  │
│ Anon (public):                     │
│ eyJhbGciOi... (token lung)        │ ← COPY THIS
│                                    │
│ Service Role (secret):             │
│ eyJhbGciOi... (token diferit)     │
└────────────────────────────────────┘
```

3. **Copy 2 valori și ține-le în safe place:**
   - **SUPABASE_URL:** `https://xxxxx.supabase.co`
   - **SUPABASE_ANON_KEY:** `eyJhbGciOi...` (public key, safe to share)

```
⚠️ DO NOT SHARE service role key!
✅ Public key (anon) e safe - o vei folosi în notebook
```

### **3.5: Run SQL Schema Setup**

1. Mergi la "SQL Editor" → New query
2. Copy-paste tot codul din `sql/supabase_setup.sql` (vom crea fișierul)
3. Click "Run"

```
Expected output:
✅ CREATE TABLE
✅ CREATE INDEX (×5)
✅ CREATE FUNCTION
```

### **3.6: Test Verify Setup**

1. Mergi la "Table Editor"
2. Trebuie să vezi tabel nou: `document_embeddings`

```
Columns:
- id (uuid, primary key)
- chunk_id (text)
- text (text)
- embedding (vector, dimension 768)
- source_pdf (text)
- page_num (integer)
- clasa (integer)
- materie (text)
- capitol (text)
- chunk_hash (text)
- has_images (boolean)
- created_at (timestamp)
```

✅ **Setup complet!**

---

# **PAP 4: CONFIGURE KAGGLE NOTEBOOK** (5 min)

### **4.1: Copy Secrets în Kaggle**

1. Mergi la [kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Scroll down la "API tokens"
3. Click "Add new API token"

```
⚠️ Asta creează kaggle.json local
Nu-l folosi pentru secrets - doar pentru CLI auth
```

4. Mergi la notebook-ul tău Kaggle
5. Click "Settings" (icon roată, dreapta-sus)
6. Tab "Secrets"

```
┌───────────────────────────────┐
│ Add secrets                   │
│                               │
│ Secret name: [____________]   │
│ Secret value: [____________]  │
│ [Add secret]                  │
└───────────────────────────────┘
```

7. Adaugă 2 secrets:

```
Secret 1:
Name: SUPABASE_URL
Value: https://xxxxx.supabase.co

Secret 2:
Name: SUPABASE_ANON_KEY
Value: eyJhbGciOi...
```

✅ **Secrets are now accessible în notebook cu:**
```python
from kaggle_secrets import UserSecretsClient
secret = UserSecretsClient()
SUPABASE_URL = secret.get_secret('SUPABASE_URL')
SUPABASE_KEY = secret.get_secret('SUPABASE_ANON_KEY')
```

### **4.2: Copy-Paste Notebook**

1. Merge la [kaggle.com/code/create](https://kaggle.com/code/create)
2. Create New Notebook
3. Copy tot codul din `kaggle_notebook.ipynb`
4. Paste în notebook

5. Configurează settings:
   - Accelerator: **GPU (P100)**
   - Internet: **Enable**
   - Persistence: **Optional**

```
Expected:
- GPU dropdown showing "GPU"
- Blue "Run All" button visible
```

### **4.3: Attach Dataset**

1. Click "+" buton (dreapta, lânga Input)
2. Search: `materiale-didactice-0-4` (sau dataset name-ul tău)
3. Click pentru attach

```
Notebook inputs (stânga):
✅ materiale-didactice-0-4
```

Notebook va accesa files la:
```python
import os
pdf_folder = '/kaggle/input/materiale-didactice-0-4/'
print(os.listdir(pdf_folder))  # Should show clasa_0, clasa_1, ...
```

---

# **PAP 5: RUN TESTE SAMPLE** (30 min)

### **5.1: Local Testing (înainte de Kaggle)**

Rulează test pe laptop cu 2-3 PDFs mici:

```bash
# Terminal/PowerShell
cd c:\Users\Opaop\Desktop\PDF-to-Embedding

# Install dependencies
pip install -r config/requirements.txt

# Run test
python tests/test_sample.py
```

Expected output:
```
Loading config...
Testing PDF extraction on 3 sample PDFs...

Processing: A1367.pdf
  ✅ Extracted 1,250 chars
  ✅ Found 5 images
  ✅ OCR processed 3 diagrams

Processing: Biblia_Romania.pdf
  ✅ Extracted 3,200 chars
  ✅ Found 2 images
  ✅ OCR processed 2 diagrams

Processing: test_manual.pdf
  ✅ Extracted 890 chars
  ✅ Found 0 images
  ✅ OCR processed 0 diagrams

Chunking & deduplication:
  ✅ Created 620 chunks
  ✅ Removed 12 duplicates (footers/headers)
  ✅ Final chunks: 608

Embedding generation:
  ✅ Generated 608 vectors (768 dimensions)
  ✅ Time: 23 seconds

Supabase connection test:
  ✅ Connected successfully
  ✅ Inserted 608 test vectors

Similarity search test:
  Query: "Cum se calculează aria unui pătrat?"
  ✅ Found 5 results
    1. "Aria = latura × latura" (similarity: 0.87)
    2. "Pătrat: 4 laturi egale" (similarity: 0.81)
    3. "Exercițiu: calculează aria pentru l=5" (similarity: 0.73)

✅ ALL TESTS PASSED!
Test vectors cleared from Supabase.
```

❌ **ERROR?** Verifică:
- [ ] Dependencies installed? (`pip install -r config/requirements.txt`)
- [ ] Supabase secrets configured? (check `test_sample.py` line 20)
- [ ] Python 3.9+? (`python --version`)

### **5.2: Kaggle Notebook Test**

1. Merge la notebook-ul tău Kaggle
2. Modifică prima celulă să proceseze doar 10 PDFs (test):

```python
# Instead of:
all_pdfs = get_all_pdfs_recursive(pdf_folder)

# Use:
all_pdfs = get_all_pdfs_recursive(pdf_folder)[:10]  # Only first 10 PDFs
```

3. Click "Run All"
4. Monitorizează output:

```
Cell 1: ✅ Setup & Dependencies
  - PyMuPDF installed
  - PaddleOCR installed
  - GPU available: Tesla P100

Cell 2: ✅ Configuration loaded

Cell 3: ✅ Found 10 PDFs (test run)
  - Total pages: ~450
  - Total size: ~125 MB

Cell 4: Processing... [████████░░░░░░░░░░░░] 40% - ETA 5 min
  - Processed PDFs: 4/10
  - Chunks generated: ~2,100
  - Vectors uploaded: 2,050/2,100

Cell 5: ✅ Post-processing complete
  - HNSW index created
  - Total vectors: 2,100
  - Database size: ~3MB

Cell 6: ✅ Validation passed
  - Sample query tested
  - Retrieval quality: excellent
```

✅ **SUCCES! Setup complet. Acum poti procesa full 15GB.**

---

# **PAP 6: FULL PROCESSING RUN** (18-24 ore)

### **6.1: Configure for Full Dataset**

1. Deschide notebook-ul Kaggle
2. Modifică celula config:

```python
# Remove the [:10] slice from previous test
all_pdfs = get_all_pdfs_recursive(pdf_folder)  # ALL PDFs

# Adjust batch sizes for 15GB (optional, poate lăsa default)
BATCH_SIZE = 10000  # Supabase batch upload
EMBEDDING_BATCH = 128  # Sentence-transformers batch
```

3. Save notebook

### **6.2: Run Full Processing**

1. Click "Run All"
2. **Lasă să ruleze overnight** (~18-24 ore)

```
Expected timeline:
- Hour 1-2: PDF extraction + OCR
- Hour 3-8: Embedding generation
- Hour 9-12: Supabase upload + indexing
- Hour 13-18: Remaining PDFs + final index creation
- Hour 19-24: Verification & cleanup
```

3. **Monitoring:**
   - Notebook auto-saves progress
   - Check output log pentru current status
   - Supabase dashboard ar trebui să arate increasing vector count

```
Dashboard statistics:
- Storage: 5MB → 50MB → 150MB → ... → 500MB
- Rows in document_embeddings: 0 → 50k → 150k → ... → 600k
```

### **6.3: Verify Final Output**

După ce notebook-ul termină:

1. Check Supabase dashboard:

```
SQL Editor query:
SELECT COUNT(*) as total_vectors FROM document_embeddings;

Expected: ~400k-600k rows
```

2. Check index status:

```sql
SELECT schemaname, tablename, indexname
FROM pg_indexes
WHERE tablename = 'document_embeddings';

Expected: HNSW index created
```

3. Test similarity search:

```sql
SELECT match_documents(
  query_embedding :=
    ARRAY[0.1, 0.2, 0.3, ... (768 values)]::float4[],
  match_count := 5,
  filter_clasa := 1
);

Expected: Top 5 similar chunks returned with similarity scores
```

✅ **DONE! Embeddings ready for AI tutoring system!**

---

# **TROUBLESHOOTING: Common Issues**

### ❌ "GPU not available" în Kaggle

**Causă:** Phone verification not done

**Soluție:**
1. Revino la **Pas 1.2** - Phone Verification
2. Restart notebook (Settings → Restart kernel)
3. Recheck GPU availability

### ❌ "Out of memory" la processing

**Causă:** Batch size prea mare

**Soluție:**
```python
# În config.yaml, reduce:
embeddings:
  batch_size: 64  # (was 128)

supabase:
  batch_size: 5000  # (was 10000)
```

### ❌ "Connection timeout" la Supabase

**Causă:** Network issues sau invalid credentials

**Soluție:**
```python
# Verify secrets in notebook:
from kaggle_secrets import UserSecretsClient
secret = UserSecretsClient()
print(secret.get_secret('SUPABASE_URL'))  # Should print URL, not error
print(secret.get_secret('SUPABASE_ANON_KEY')[:20])  # First 20 chars of key

# If error: check Settings → Secrets on Kaggle
```

### ❌ "PDF parsing fails for some files"

**Causă:** Corrupted PDFs sau format incompatibil

**Soluție:**
```python
# Error handling in notebook automatically skips bad PDFs:
try:
    text = extract_text_and_images(pdf_path)
except:
    logger.warning(f"Skipped {pdf_path} - corrupted")
    continue  # Move to next PDF

# Final report will show:
# ✅ Successfully processed: 4,950
# ⚠️ Skipped (corrupted): 45
```

---

# **FINAL CHECKLIST**

Odată ce setup e complet:

- [x] Kaggle account cu phone verification
- [x] GPU (P100) available în notebook
- [x] Dataset uploaded (15GB PDFs)
- [x] Supabase project created
- [x] pgvector extension enabled
- [x] SQL schema deployed
- [x] API keys copied to notebook secrets
- [x] Sample test passed locally
- [x] Sample test passed în Kaggle (10 PDFs)
- [x] Full processing notebook configured

✅ **YOU'RE READY TO PROCESS 15GB OF MANUALS!**

Rulează "Run All" și lasă să proceseze overnight.

---

**Next:** [ARCHITECTURE.md](docs/ARCHITECTURE.md) pentru înțelegere tehnică
