-- supabase_setup.sql
-- Schema și indexes pentru pgvector embeddings storage în Supabase
--
-- Instrucțiuni:
-- 1. Deschide Supabase SQL Editor
-- 2. Copy-paste tot codul din acest fișier
-- 3. Click "Run"
-- 4. Verifică: table și indexes create successful

-- ==============================================================================
-- 1. ENABLE pgvector EXTENSION
-- ==============================================================================
-- Dacă nu e încă enabled, activează extensia pgvector
CREATE EXTENSION IF NOT EXISTS vector;


-- ==============================================================================
-- 2. CREATE MAIN TABLE: document_embeddings
-- ==============================================================================
-- Tabel principal pentru stocarea embeddings din PDF-uri
CREATE TABLE IF NOT EXISTS document_embeddings (
    -- Identificatori
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id TEXT UNIQUE NOT NULL,  -- Identificator chunk unic

    -- Content
    text TEXT NOT NULL,  -- Text chunk (max 10000 chars)
    embedding VECTOR(768) NOT NULL,  -- Vector embedding 768-dimensional

    -- Source metadata
    source_pdf TEXT NOT NULL,  -- Path PDF source (ex: clasa_1/matematica.pdf)
    page_num INTEGER,  -- Pagina din PDF

    -- Educational metadata
    clasa INTEGER,  -- Clasa (0-4)
    materie TEXT,  -- Materie (Matematică, Română, etc)
    capitol TEXT,  -- Capitol din manual

    -- Technical metadata
    chunk_hash TEXT UNIQUE,  -- MD5 hash pentru deduplicare
    has_images BOOLEAN DEFAULT FALSE,  -- Dacă chunk conținea imagini

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Comentariu pentru descriere tabel
COMMENT ON TABLE document_embeddings IS 'Embeddings din manualele PDF procesate cu sentence-transformers';
COMMENT ON COLUMN document_embeddings.embedding IS 'Vector 768-dimensional din model paraphrase-multilingual-mpnet-base-v2';
COMMENT ON COLUMN document_embeddings.chunk_hash IS 'MD5 hash pentru identificare duplicates (headers/footers)';


-- ==============================================================================
-- 3. CREATE INDEXES FOR FAST QUERIES
-- ==============================================================================

-- Index HNSW pentru similarity search (cosine distance)
-- Aceasta permite căutări rapide de vectori similari (~50ms pentru 600k vectors)
CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
ON document_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index pentru filtrare pe clasa
CREATE INDEX IF NOT EXISTS idx_clasa
ON document_embeddings(clasa);

-- Index pentru filtrare pe materie
CREATE INDEX IF NOT EXISTS idx_materie
ON document_embeddings(materie);

-- Index pentru căutări pe source_pdf
CREATE INDEX IF NOT EXISTS idx_source_pdf
ON document_embeddings(source_pdf);

-- Index pentru căutări pe chunk_id (unique, dar explicit index e bun)
CREATE INDEX IF NOT EXISTS idx_chunk_id
ON document_embeddings(chunk_id);

-- Index pe created_at (pentru timeseries queries)
CREATE INDEX IF NOT EXISTS idx_created_at
ON document_embeddings(created_at DESC);


-- ==============================================================================
-- 4. CREATE RPC FUNCTION: match_documents
-- ==============================================================================
-- Funcție pentru similarity search cu filtrare pe clasa/materie
-- Utilizare din client:
--   uploader.query_similar(query_embedding, match_count=5, filter_clasa=1)
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(768),
    match_count INT DEFAULT 5,
    filter_clasa INT DEFAULT NULL,
    filter_materie TEXT DEFAULT NULL
)
RETURNS TABLE (
    chunk_id TEXT,
    text TEXT,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.chunk_id,
        d.text,
        (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity,
        jsonb_build_object(
            'source_pdf', d.source_pdf,
            'page_num', d.page_num,
            'clasa', d.clasa,
            'materie', d.materie,
            'capitol', d.capitol,
            'has_images', d.has_images,
            'created_at', d.created_at
        ) AS metadata
    FROM document_embeddings d
    WHERE
        (filter_clasa IS NULL OR d.clasa = filter_clasa)
        AND (filter_materie IS NULL OR d.materie = filter_materie)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_documents(VECTOR, INT, INT, TEXT) IS 'Similarity search pentru embeddings cu filtrare pe clasa/materie';


-- ==============================================================================
-- 5. CREATE RPC FUNCTION: get_statistics
-- ==============================================================================
-- Obține statistici despre database
CREATE OR REPLACE FUNCTION get_statistics()
RETURNS TABLE (
    total_vectors BIGINT,
    unique_pdfs BIGINT,
    clasa_distribution TEXT,
    oldest_entry TIMESTAMP,
    newest_entry TIMESTAMP
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT,
        COUNT(DISTINCT source_pdf)::BIGINT,
        json_object_agg(
            COALESCE(clasa::TEXT, 'unknown'),
            COUNT(*)
        )::TEXT,
        MIN(created_at),
        MAX(created_at)
    FROM document_embeddings;
END;
$$;

COMMENT ON FUNCTION get_statistics() IS 'Returnează statistici generale despre embeddings';


-- ==============================================================================
-- 6. CREATE RPC FUNCTION: count_vectors
-- ==============================================================================
-- Rapid count pentru progress tracking
CREATE OR REPLACE FUNCTION count_vectors()
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
    total_count INT;
BEGIN
    SELECT COUNT(*)::INT INTO total_count
    FROM document_embeddings;
    RETURN total_count;
END;
$$;

COMMENT ON FUNCTION count_vectors() IS 'Rapid vector count';


-- ==============================================================================
-- 7. CREATE VIEW: embeddings_summary
-- ==============================================================================
-- View pentru overview rapid
CREATE OR REPLACE VIEW embeddings_summary AS
SELECT
    COUNT(*) as total_vectors,
    COUNT(DISTINCT source_pdf) as unique_pdfs,
    COUNT(DISTINCT clasa) as unique_classes,
    COUNT(DISTINCT materie) as unique_subjects,
    MIN(created_at) as earliest,
    MAX(created_at) as latest,
    (SELECT COUNT(*) FILTER (WHERE has_images = TRUE)) as vectors_with_images
FROM document_embeddings;

COMMENT ON VIEW embeddings_summary IS 'Summary statistici pentru tabel embeddings';


-- ==============================================================================
-- 8. ENABLE ROW LEVEL SECURITY (OPTIONAL)
-- ==============================================================================
-- Pentru Supabase, RLS e util dar nu necesar pentru free tier public access
-- Uncomment dacă vrei să restricționezi accesul

/*
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;

-- Policy: Allow all to read (public access)
CREATE POLICY "Allow public read access"
    ON document_embeddings
    FOR SELECT
    TO public
    USING (true);

-- Policy: Allow inserts din service role
CREATE POLICY "Allow service role insert"
    ON document_embeddings
    FOR INSERT
    TO authenticated
    WITH CHECK (true);
*/


-- ==============================================================================
-- 9. VERIFICATION QUERIES
-- ==============================================================================
-- Rulează aceste după setup pentru a verifica totul e OK

-- Check extension
-- SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check table exists
-- SELECT table_name FROM information_schema.tables WHERE table_name = 'document_embeddings';

-- Check indexes created
-- SELECT indexname FROM pg_indexes WHERE tablename = 'document_embeddings';

-- Check functions exist
-- SELECT proname FROM pg_proc WHERE proname IN ('match_documents', 'get_statistics', 'count_vectors');

-- ==============================================================================
-- DONE!
-- ==============================================================================
-- Schema-ul e gata. Poți acum:
-- 1. Upload vectors cu supabase_uploader.py
-- 2. Query cu match_documents RPC function
-- 3. Get stats cu get_statistics() function
--
-- Exemple queries (din SQL Editor sau programmatic):
--
-- SELECT COUNT(*) FROM document_embeddings;  -- Total vectors
--
-- SELECT * FROM get_statistics();  -- Statistici complete
--
-- SELECT COUNT_vectors();  -- Rapid count
--
-- SELECT * FROM embeddings_summary;  -- View summary
--
-- SELECT * FROM match_documents(
--     query_embedding := '[0.1, 0.2, ..., 0.x]'::vector,
--     match_count := 5,
--     filter_clasa := 1
-- );  -- Similarity search
