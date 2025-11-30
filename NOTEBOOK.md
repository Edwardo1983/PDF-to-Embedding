# Ghid rulare NOTEBOOK.ipynb (Kaggle si local)

Acest ghid este pas cu pas pentru a rula `NOTEBOOK.ipynb` folosind codul din `scripts/`. Exemplele sunt gandite pentru Kaggle, dar functioneaza si local.

## Ordinea celulelor din NOTEBOOK.ipynb
1) Setup & Dependencies — instaleaza pachetele (pip -q).
2) Import Modules & Configuration — loc pentru importuri/config (ex. USE_GPU, MAX_PAGES).
3) PDF Extractor — cod din `scripts/pdf_extractor.py`.
4) OCR Processor — cod din `scripts/ocr_processor.py`.
5) Chunking Utilities — cod din `scripts/chunking.py`.
6) Embedding Generator — cod din `scripts/embedding_generator.py`.
7) Supabase Uploader — cod din `scripts/supabase_uploader.py`.
8) DONE! — mesaj final.

## Pas cu pas (Kaggle)
1) **Adauga datasetul cu PDF-urile** (butonul Add data in notebook). Asigura-te ca fisierul dorit exista, de ex. `A1367.pdf` in structura originala.
2) **Ruleaza Celula 1** (install). Daca vezi mici conflicte la pip, lasa-l sa termine; la final trebuie sa spuna ca a instalat cu succes.
3) **Celula 2 (Import/Config)**: pune configurari minime, exemplu:
   ```python
   from scripts import chunking, embedding_generator, ocr_processor, pdf_extractor, supabase_uploader
   USE_GPU = False
   MAX_PAGES = 2  # creste daca vrei intregul PDF
   BATCH_SIZE = 64
   ```
4) **Celula 3 (PDF Extractor)**: seteaza calea reala a PDF-ului (inlocuieste `test_sample.pdf`):
   ```python
   pdf_path = "/kaggle/input/<dataset-name>/materiale_didactice/Scoala_de_Muzica_George_Enescu/clasa_1/Comunicare_in_Limba_Romana/Prof_Ion_Creanga/A1367.pdf"
   result = extract_text_and_images(pdf_path, max_pages=MAX_PAGES)
   print(result.extraction_status, len(result.text), "chars", len(result.images), "images")
   ```
   Daca folosesti alt PDF, schimba doar `pdf_path`.
5) **Celula 4 (OCR Processor, optional)**: ruleaza numai daca PDF-ul are imagini cu text sau daca `result.text` e gol.
   ```python
   processor = ocr_processor.OCRProcessor(languages=["ro", "en"], use_gpu=USE_GPU)
   # Exemplu minimal: proceseaza imaginile cu prioritate mare
   processed_images = processor.process_images_selective(
       [{"image_array": img.get("image_array"), "priority_score": 1.0} for img in []],  # inlocuieste cu imaginile reale
       priority_threshold=0.5
   )
   ocr_text = "\n".join(img.get("ocr_text", "") for img in processed_images)
   ```
   Daca nu ai nevoie de OCR, sari peste celula sau ruleaz-o asa cum este.
6) **Celula 5 (Chunking)**: foloseste textul din extractor sau din OCR (daca extractorul nu a dat text).
   ```python
   text_to_chunk = result.text.strip() or ocr_text
   chunker = chunking.TextChunker(chunk_size=400, overlap=40, min_chunk_length=80)
   chunks = chunker.chunk_text(text_to_chunk, source_page=1, remove_duplicates=True)
   print(len(chunks), "chunks")
   ```
7) **Celula 6 (Embedding Generator)**: genereaza embeddings; pe Kaggle GPU seteaza `USE_GPU = True` daca vrei.
   ```python
   generator = embedding_generator.EmbeddingGenerator(device="cuda" if USE_GPU else "cpu", batch_size=BATCH_SIZE)
   texts = [c.text for c in chunks]
   embeddings = generator.generate_embeddings(texts, show_progress=False)
   print(len(embeddings), "embeddings")
   ```
8) **Celula 7 (Supabase Uploader, optional)**: ruleaza doar daca vrei upload.
   ```python
   import os
   os.environ["SUPABASE_URL"] = "https://xxxx.supabase.co"
   os.environ["SUPABASE_ANON_KEY"] = "<key>"
   uploader = supabase_uploader.SupabaseUploader(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
   uploader.connect()
   # pregateste vectors_data conform supabase_uploader docs; apoi:
   # uploader.upload_vectors(vectors_data, show_progress=True)
   ```
   Daca nu vrei upload, sari peste celula.

## Pas cu pas (local)
1) Creeaza si activeaza un venv (recomandat), apoi `pip install -r requirements.txt`.
2) Ruleaza notebook-ul: `jupyter notebook NOTEBOOK.ipynb` sau in VSCode.
3) Seteaza `pdf_path` catre un PDF existent local; restul pasilor sunt la fel.

## Curatare / spatiu
- Pe Kaggle, `.venv` nu se foloseste; poti ignora.
- Local, poti sterge `.venv` si cache-urile dupa ce ai generat `.ipynb` daca vrei spatiu.

## Push pe GitHub
- Daca vrei sa publici schimbarile:
  ```bash
  git add build_notebook.py NOTEBOOK.ipynb NOTEBOOK.md
  git commit -m "Improve notebook guide and generator"
  git push origin main
  ```
  (Comenzile nu au fost rulate aici.)
