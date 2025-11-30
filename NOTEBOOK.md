# Ghid rulare NOTEBOOK.ipynb (Kaggle și local) – pas cu pas, pentru începători

Acest ghid te ajută să rulezi `NOTEBOOK.ipynb` folosind codul din `scripts/`. Exemplele sunt pentru Kaggle, dar merg și local.

## Ordinea celulelor
1) Setup & Dependencies — instalează pachetele (pip -q).
2) Import Modules & Configuration — importuri/config (ex. USE_GPU, MAX_PAGES, BATCH_SIZE).
3) PDF Extractor — cod din `scripts/pdf_extractor.py`.
4) OCR Processor — cod din `scripts/ocr_processor.py` (obligatoriu dacă PDF-ul are imagini sau text mic de tip imagine).
5) Chunking Utilities — cod din `scripts/chunking.py`.
6) Embedding Generator — cod din `scripts/embedding_generator.py`.
7) Supabase Uploader — cod din `scripts/supabase_uploader.py`.
8) DONE! — mesaj final.

## Parametri cheie (ce înseamnă și când să îi schimbi)
- `USE_GPU` (True/False): dacă ai GPU disponibil (Kaggle P100/T4). True = folosește GPU (mai rapid, dar consumă memorie), False = CPU (mai lent, dar stabil).
- `MAX_PAGES`: câte pagini să procesezi din PDF (pentru debug). Pune 2–5 la test, sau `None`/valoare mare pentru tot PDF-ul.
- `BATCH_SIZE`: câte texte procesezi odată la embeddings. Mai mare = mai rapid pe GPU, dar consumă memorie. Pentru CPU rămâi la 32–64; pe GPU poți crește (128).
- Chunking: `chunk_size` = lungimea în caractere a unui fragment de text; `overlap` = câte caractere se repetă între fragmente (context); `min_chunk_length` = fragmente mai scurte de atât sunt ignorate. Mai mare = mai puține chunk-uri, dar mai „grele”; mai mic = mai multe chunk-uri (cost embeddings mai mare).

## Pas cu pas (Kaggle)
1) **Adaugă datasetul cu PDF-uri** (butonul Add data). Verifică să ai `A1367.pdf` la calea originală:
   `/kaggle/input/<dataset-name>/materiale_didactice/Scoala_de_Muzica_George_Enescu/clasa_1/Comunicare_in_Limba_Romana/Prof_Ion_Creanga/A1367.pdf`
2) **Rulează Celula 1**. Instalează dependențele. Lasă pip să termine chiar dacă vezi mesaje mixte; la final trebuie să fie ok.
3) **Celula 2 (Import/Config)**: setează parametrii de bază, explicat:
   ```python
   from scripts import chunking, embedding_generator, ocr_processor, pdf_extractor, supabase_uploader
   USE_GPU = False          # True dacă vrei GPU (mai rapid), False dacă vrei CPU (mai stabil)
   MAX_PAGES = 2            # pune 2-5 pentru test; crește dacă vrei tot PDF-ul
   BATCH_SIZE = 64          # 32-64 pe CPU; poți crește pe GPU dacă ai memorie
   ```
4) **Celula 3 (PDF Extractor)**: pune calea PDF-ului real, nu `test_sample.pdf`:
   ```python
   pdf_path = "/kaggle/input/<dataset-name>/materiale_didactice/Scoala_de_Muzica_George_Enescu/clasa_1/Comunicare_in_Limba_Romana/Prof_Ion_Creanga/A1367.pdf"
   result = extract_text_and_images(pdf_path, max_pages=MAX_PAGES)
   print(result.extraction_status, len(result.text), "chars", len(result.images), "images")
   ```
   - Dacă vrei alt PDF, schimbă doar `pdf_path`.
5) **Celula 4 (OCR Processor, obligatoriu dacă PDF-ul are imagini)**:
   - PDF-urile scanate sau cu diagrame au text în imagini; atunci ai nevoie de OCR.
   - Folosește imaginile extrase de extractor (din `result.images`). Dacă vrei să procesezi toate imaginile extrase cu prioritate ≥0.5:
     ```python
     processor = ocr_processor.OCRProcessor(languages=["ro", "en"], use_gpu=USE_GPU)
     # Aici trebuie efectiv imaginile (în obiecte Pixmap / arrays). Dacă extractorul tău returnează imaginea ca bytes/array,
     # construiește lista așa:
     images_for_ocr = []
     for img in result.images:
         # img trebuie să aibă 'image_array' (np.ndarray). Dacă extractorul tău nu atașează array-ul,
         # trebuie să îl obții manual din PDF (ex. cu PyMuPDF pixmap -> np array).
         if img.get("image_array") is not None and img.get("priority_score", 0) >= 0.5:
             images_for_ocr.append({"image_array": img["image_array"], "priority_score": img.get("priority_score", 0.5)})

     processed_images = processor.process_images_selective(images_for_ocr, priority_threshold=0.5)
     ocr_text = "\n".join(img.get("ocr_text", "") for img in processed_images)
     print("OCR chars:", len(ocr_text))
     ```
   - Dacă PDF-ul este doar text și ai `result.text` consistent, OCR poate fi ocolit, dar pentru scanări e obligatoriu ca să obții text.
6) **Celula 5 (Chunking)**: fragmentează textul pentru embeddings.
   ```python
   text_to_chunk = result.text.strip()
   if not text_to_chunk:  # dacă extractorul nu a produs text (imagini doar), folosește textul din OCR
       text_to_chunk = ocr_text
   chunker = chunking.TextChunker(chunk_size=400, overlap=40, min_chunk_length=80)
   chunks = chunker.chunk_text(text_to_chunk, source_page=1, remove_duplicates=True)
   print(len(chunks), "chunks")
   ```
   - Dacă mărești `chunk_size`, vei avea mai puține chunk-uri (dar mai lungi). Dacă micșorezi, vei avea mai multe chunk-uri (cost embeddings mai mare).
7) **Celula 6 (Embedding Generator)**: calculează embeddings.
   ```python
   generator = embedding_generator.EmbeddingGenerator(
       device="cuda" if USE_GPU else "cpu",
       batch_size=BATCH_SIZE
   )
   texts = [c.text for c in chunks]
   embeddings = generator.generate_embeddings(texts, show_progress=False)
   print(len(embeddings), "embeddings")
   ```
   - `USE_GPU=True` folosește GPU (dacă există) — mai rapid, consum mai mare de memorie. False = CPU (mai lent, stabil).
8) **Celula 7 (Supabase Uploader)**: scopul final, încarcă embeddings în Supabase.
   ```python
   import os
   os.environ["SUPABASE_URL"] = "https://xxxx.supabase.co"
   os.environ["SUPABASE_ANON_KEY"] = "<key>"

   uploader = supabase_uploader.SupabaseUploader(
       os.environ["SUPABASE_URL"],
       os.environ["SUPABASE_ANON_KEY"]
   )
   uploader.connect()

   # Construiește vectors_data pe baza chunks + embeddings
   vectors_data = []
   for i, emb in enumerate(embeddings):
       vectors_data.append({
           "chunk_id": chunks[i].chunk_id if hasattr(chunks[i], "chunk_id") else f"chunk_{i}",
           "text": chunks[i].text,
           "embedding": emb.tolist() if hasattr(emb, "tolist") else emb,
           "source_pdf": pdf_path,
           "page_num": getattr(chunks[i], "source_page", 0),
           "clasa": 1,
           "materie": "Romana",
           "capitol": "Comunicare",
           "chunk_hash": getattr(chunks[i], "chunk_hash", ""),
           "has_images": len(getattr(result, "images", [])) > 0
       })

   stats = uploader.upload_vectors(vectors_data, show_progress=True)
   print("Upload stats:", stats)
   ```
   - Nu expune cheile în notebook public; folosește variabile de mediu în privat.

## Pas cu pas (local)
1) Creează/activează venv, `pip install -r requirements.txt`.
2) Rulează notebook-ul: `jupyter notebook NOTEBOOK.ipynb` sau VSCode.
3) Setează `pdf_path` spre un PDF existent local; restul pasilor sunt identici.

## Curățare / spațiu
- Pe Kaggle, `.venv` nu se folosește; poți ignora.
- Local, poți șterge `.venv` și cache-urile după ce ai generat `.ipynb` dacă vrei spațiu.

## Push pe GitHub
- (opțional) Publică schimbările:
  ```bash
  git add build_notebook.py NOTEBOOK.ipynb NOTEBOOK.md
  git commit -m "Improve notebook guide and generator"
  git push origin main
  ```
  (în acest proiect commit/push a fost deja făcut în schimbarea precedentă; aceste comenzi sunt doar de referință)
