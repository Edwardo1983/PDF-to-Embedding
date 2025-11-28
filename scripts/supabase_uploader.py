"""
supabase_uploader.py - Batch upload de embeddings în Supabase pgvector

Implementează:
1. Connection pooling la Supabase
2. Batch insert (10k vectors per batch)
3. Retry logic (3 attempts per batch)
4. Progress tracking
5. Verification post-upload

Supabase pgvector:
- 500MB free tier permanent
- HNSW indexing pentru fast similarity search
- RPC functions pentru query
"""

import logging
from typing import List, Dict, Optional, Tuple
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseUploader:
    """
    Uploader pentru vectors în Supabase pgvector.

    Utilizare:
        uploader = SupabaseUploader(supabase_url, supabase_key)
        uploader.connect()
        uploader.upload_vectors(vectors, batch_size=10000)
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_anon_key: str,
        batch_size: int = 10000,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        """
        Inițializează uploader.

        Args:
            supabase_url: Project URL (ex: https://xxxxx.supabase.co)
            supabase_anon_key: Anon/public API key
            batch_size: Vectori per batch insert
            max_retries: Retry attempts pe batch
            retry_delay: Delay între retries (secunde)
        """

        self.supabase_url = supabase_url
        self.supabase_key = supabase_anon_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = None
        self.table_name = "document_embeddings"

        logger.info(f"SupabaseUploader initialized")
        logger.info(f"Batch size: {batch_size}, Max retries: {max_retries}")

    def connect(self) -> bool:
        """
        Conectează la Supabase.

        Returns:
            True dacă conexiunea e reușită, False altfel
        """

        try:
            from supabase import create_client

            logger.info(f"Connecting to Supabase: {self.supabase_url}")

            self.client = create_client(self.supabase_url, self.supabase_key)

            # Test conexiune
            response = self.client.table(self.table_name).select("*").limit(1).execute()

            logger.info("✅ Successfully connected to Supabase")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            return False

    def upload_vectors(
        self,
        vectors_data: List[Dict],
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Upload batch de vectors în Supabase.

        Format vectors_data:
        [
            {
                'chunk_id': 'chunk_001',
                'text': 'Aria pătrat...',
                'embedding': [0.1, 0.2, ...],  # lista 768 valores
                'source_pdf': 'clasa_1/matematica.pdf',
                'page_num': 5,
                'clasa': 1,
                'materie': 'Matematică',
                'capitol': 'Geometrie',
                'chunk_hash': 'abc123...',
                'has_images': True
            },
            ...
        ]

        Args:
            vectors_data: Lista de dicts cu vector info
            show_progress: Să arăți progress bar

        Returns:
            Dict cu statistici: {'success': N, 'failed': M}
        """

        if not self.client:
            logger.error("Not connected to Supabase. Call .connect() first.")
            return {'success': 0, 'failed': 0}

        if not vectors_data:
            logger.warning("Empty vectors list")
            return {'success': 0, 'failed': 0}

        logger.info(f"Uploading {len(vectors_data)} vectors...")

        success_count = 0
        failed_count = 0

        # Progres tracking
        if show_progress:
            try:
                from tqdm import tqdm
                batches = [
                    vectors_data[i:i + self.batch_size]
                    for i in range(0, len(vectors_data), self.batch_size)
                ]
                iterator = tqdm(batches, desc="Uploading batches")
            except ImportError:
                batches = [
                    vectors_data[i:i + self.batch_size]
                    for i in range(0, len(vectors_data), self.batch_size)
                ]
                iterator = batches
        else:
            batches = [
                vectors_data[i:i + self.batch_size]
                for i in range(0, len(vectors_data), self.batch_size)
            ]
            iterator = batches

        # Upload fiecare batch
        for batch in iterator:
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    # Formatează data pentru Supabase
                    formatted_batch = self._format_batch(batch)

                    # Insert în database
                    response = self.client.table(self.table_name).insert(
                        formatted_batch
                    ).execute()

                    success_count += len(batch)
                    logger.debug(f"Batch uploaded successfully ({len(batch)} rows)")
                    break  # Succes, pass la următorul batch

                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                    )

                    if attempt == self.max_retries - 1:
                        # Ultima încercare eșuată
                        failed_count += len(batch)
                        logger.error(f"Batch upload failed after {self.max_retries} attempts")
                    else:
                        # Retry cu delay
                        time.sleep(self.retry_delay)

        logger.info(
            f"Upload complete: {success_count} success, {failed_count} failed"
        )

        return {'success': success_count, 'failed': failed_count}

    def _format_batch(self, vectors_data: List[Dict]) -> List[Dict]:
        """
        Formatează batch de vectors pentru insert în Supabase.

        Conversii:
        - embedding: list → string JSON (pgvector format)
        - Validare camp-uri necesare

        Args:
            vectors_data: Raw vector data

        Returns:
            Formatted data pentru Supabase insert
        """

        formatted = []

        for item in vectors_data:
            # Validare camp-uri obligatorii
            required = ['chunk_id', 'text', 'embedding', 'source_pdf', 'page_num', 'clasa']

            if not all(key in item for key in required):
                logger.warning(f"Skipping invalid item: missing required fields")
                continue

            # Format embedding (list → vector type)
            embedding = item['embedding']
            if isinstance(embedding, list):
                embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
            else:
                embedding_str = str(embedding)

            formatted_item = {
                'chunk_id': item['chunk_id'],
                'text': item['text'][:10000],  # Limit text la 10k chars
                'embedding': embedding_str,
                'source_pdf': item.get('source_pdf', 'unknown'),
                'page_num': int(item.get('page_num', 0)),
                'clasa': int(item.get('clasa', 0)),
                'materie': item.get('materie', 'unknown'),
                'capitol': item.get('capitol', 'unknown'),
                'chunk_hash': item.get('chunk_hash', ''),
                'has_images': bool(item.get('has_images', False))
            }

            formatted.append(formatted_item)

        return formatted

    def verify_upload(self, expected_count: int) -> Tuple[bool, int]:
        """
        Verifică că toti vectorii au fost uploaduiți.

        Args:
            expected_count: Numărul așteptat de vectori

        Returns:
            (success: bool, actual_count: int)
        """

        try:
            response = self.client.table(self.table_name).select("count", count="exact").execute()

            actual_count = response.count
            logger.info(f"Verification: expected {expected_count}, actual {actual_count}")

            if actual_count >= expected_count:
                logger.info("✅ Upload verification passed")
                return True, actual_count
            else:
                logger.warning(f"⚠️ Fewer vectors than expected")
                return False, actual_count

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False, -1

    def query_similar(
        self,
        query_embedding: List[float],
        match_count: int = 5,
        filter_clasa: Optional[int] = None,
        filter_materie: Optional[str] = None
    ) -> List[Dict]:
        """
        Query pentru similar vectors (semantic search).

        Args:
            query_embedding: Query embedding (768-dim)
            match_count: Câți vectori să return
            filter_clasa: (Optional) Filter pe clasa
            filter_materie: (Optional) Filter pe materie

        Returns:
            Lista de rezultate similare

        Exemplu:
            >>> query_emb = [0.1, 0.2, ...]  # 768 values
            >>> results = uploader.query_similar(query_emb, match_count=5, filter_clasa=1)
            >>> for r in results:
            ...     print(f"{r['text'][:50]} - similarity: {r['similarity']:.2%}")
        """

        try:
            # Format embedding ca SQL array
            embedding_array = '[' + ','.join(str(x) for x in query_embedding) + ']'

            # Call RPC function (definit în sql/supabase_setup.sql)
            response = self.client.rpc(
                'match_documents',
                {
                    'query_embedding': embedding_array,
                    'match_count': match_count,
                    'filter_clasa': filter_clasa,
                    'filter_materie': filter_materie
                }
            ).execute()

            logger.debug(f"Query returned {len(response.data)} results")
            return response.data

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """
        Obține statistici despre database.

        Returns:
            Dict cu stats: total vectors, storage size, etc
        """

        try:
            # Total vectors
            response = self.client.table(self.table_name).select("count", count="exact").execute()
            total_vectors = response.count

            # Unique PDFs
            pdf_response = self.client.table(self.table_name).select("source_pdf").execute()
            unique_pdfs = len(set(item['source_pdf'] for item in pdf_response.data))

            stats = {
                'total_vectors': total_vectors,
                'unique_pdfs': unique_pdfs,
                'embedding_dim': 768,
                'status': 'healthy' if total_vectors > 0 else 'empty'
            }

            logger.info(f"Database stats: {total_vectors} vectors from {unique_pdfs} PDFs")
            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


def upload_vectors_batch(
    vectors_data: List[Dict],
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    batch_size: int = 10000
) -> Dict:
    """
    Utility function: upload vectors cu config automată din environment.

    Args:
        vectors_data: Vectors pentru upload
        supabase_url: (Optional) URL, sau din env SUPABASE_URL
        supabase_key: (Optional) Key, sau din env SUPABASE_ANON_KEY
        batch_size: Batch size

    Returns:
        Stats dict

    Exemplu:
        >>> vectors = [{'chunk_id': 'c1', 'embedding': [...], ...}, ...]
        >>> stats = upload_vectors_batch(vectors)
        >>> print(f"Uploaded {stats['success']} vectors")
    """

    # Obține credentials din environment
    if not supabase_url:
        supabase_url = os.getenv('SUPABASE_URL')
    if not supabase_key:
        supabase_key = os.getenv('SUPABASE_ANON_KEY')

    if not supabase_url or not supabase_key:
        logger.error("Supabase credentials not provided or not in environment")
        return {'success': 0, 'failed': 0}

    # Upload
    uploader = SupabaseUploader(supabase_url, supabase_key, batch_size=batch_size)

    if uploader.connect():
        return uploader.upload_vectors(vectors_data, show_progress=True)
    else:
        return {'success': 0, 'failed': 0}


# Exemplu de utilizare
if __name__ == "__main__":
    import os

    # Obține credentials din environment
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')

    if not supabase_url or not supabase_key:
        print("⚠️ Supabase credentials not in environment")
        print("Set SUPABASE_URL and SUPABASE_ANON_KEY to test")
    else:
        # Inițializează uploader
        uploader = SupabaseUploader(supabase_url, supabase_key)

        # Conectează
        if uploader.connect():
            # Get stats
            stats = uploader.get_database_stats()
            print(f"Database stats: {stats}")

            # Test query
            print("\nTest query not implemented (requires actual embeddings)")
