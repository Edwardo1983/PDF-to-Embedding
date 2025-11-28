"""
embedding_generator.py - Generează embeddings cu sentence-transformers

Transformă text chunks în vector embeddings 768-dimensional.

Model: paraphrase-multilingual-mpnet-base-v2
- Suportă 50+ limbi (inclusiv română)
- 768 dimensiuni (bun pentru similaritate semantică)
- Funcționează pe CPU și GPU
- Stabil pentru batch processing

Batch processing: procesează 128 chunks la o dată pentru eficiență GPU.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Rezultat al generării embedding pentru chunk"""
    text: str
    embedding: np.ndarray  # Array numpy (768,)
    dimension: int  # Should be 768
    model_name: str  # Model folosit
    generation_time_ms: float


class EmbeddingGenerator:
    """
    Generator de embeddings cu sentence-transformers.

    Utilizare:
        generator = EmbeddingGenerator(model_name='paraphrase-multilingual-mpnet-base-v2')
        embeddings = generator.generate_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings of {embeddings[0].shape[0]} dims")
    """

    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
        device: str = 'cpu',  # 'cpu' sau 'cuda' pentru GPU
        batch_size: int = 128
    ):
        """
        Inițializează embedding generator.

        Args:
            model_name: Model din sentence-transformers
            device: 'cpu' (default, mai stabil) sau 'cuda' (GPU dacă disponibil)
            batch_size: Procesare batch (dimensiune)

        Detalii model:
            paraphrase-multilingual-mpnet-base-v2:
            - 768 dimensiuni
            - ~470M parametri
            - Foarte bun pentru similaritate semantică
            - Suportă 50+ limbi
        """

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {device}")

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)

            # Obține dimensiuni
            dummy_embedding = self.model.encode(["test"])[0]
            self.dimensions = len(dummy_embedding)

            logger.info(f"✅ Model loaded successfully. Dimensions: {self.dimensions}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generează embeddings pentru o listă de texts.

        Batch processing pentru eficiență: procesează batch_size texte la o dată.

        Args:
            texts: Listă de strings pentru embedding
            show_progress: Să arăți progress bar

        Returns:
            Listă de numpy arrays (fiecare ar trebui să fie (768,))

        Exemplu:
            >>> texts = ["Aria pătrat = latura²", "Perimetru = 4 × latura", ...]
            >>> generator = EmbeddingGenerator()
            >>> embeddings = generator.generate_embeddings(texts)
            >>> print(f"Generated {len(embeddings)} embeddings")
            "Generated 1250 embeddings"
            >>> print(embeddings[0].shape)
            "(768,)"
        """

        import time

        if not texts:
            logger.warning("Empty text list provided")
            return []

        start_time = time.time()
        logger.info(f"Generating embeddings for {len(texts)} texts...")

        try:
            # Procesează în batches pentru eficiență
            all_embeddings = []

            # Progres tracking
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(
                        range(0, len(texts), self.batch_size),
                        desc="Embeddings"
                    )
                except ImportError:
                    iterator = range(0, len(texts), self.batch_size)
            else:
                iterator = range(0, len(texts), self.batch_size)

            for i in iterator:
                batch = texts[i:i + self.batch_size]

                # Generează embeddings pentru batch
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True  # Output numpy arrays
                )

                all_embeddings.extend(batch_embeddings)

            elapsed = time.time() - start_time
            rate = len(texts) / elapsed if elapsed > 0 else 0

            logger.info(
                f"✅ Generated {len(all_embeddings)} embeddings in {elapsed:.1f}s "
                f"({rate:.0f} texts/sec)"
            )

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generează embedding pentru un singur text.

        Util pentru query time (client-side similarity search).

        Args:
            text: Text pentru embedding

        Returns:
            Numpy array (768,)

        Exemplu:
            >>> query = "Cum se calculează aria unui dreptunghi?"
            >>> embedding = generator.generate_single_embedding(query)
            >>> print(embedding.shape)
            "(768,)"
        """

        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding

    def generate_embeddings_with_metadata(
        self,
        text_chunks: List[dict],  # Format: {'text': str, 'metadata': dict}
        show_progress: bool = True
    ) -> List[dict]:
        """
        Generează embeddings și prezerva metadate.

        Args:
            text_chunks: Listă de dicts cu 'text' și optional 'metadata'
            show_progress: Progress bar

        Returns:
            Lista de dicts cu embedding + original metadata

        Exemplu:
            >>> chunks = [
            ...     {'text': 'Text 1', 'metadata': {'page': 1}},
            ...     {'text': 'Text 2', 'metadata': {'page': 2}},
            ... ]
            >>> results = generator.generate_embeddings_with_metadata(chunks)
            >>> print(results[0].keys())
            dict_keys(['text', 'embedding', 'metadata', ...])
        """

        # Extrage texte
        texts = [chunk['text'] for chunk in text_chunks]

        # Generează embeddings
        embeddings = self.generate_embeddings(texts, show_progress=show_progress)

        # Recombine cu metadate
        results = []
        for i, chunk in enumerate(text_chunks):
            result = {
                'text': chunk['text'],
                'embedding': embeddings[i],
                'metadata': chunk.get('metadata', {}),
                'embedding_dim': len(embeddings[i])
            }
            results.append(result)

        return results

    def get_model_info(self) -> dict:
        """
        Returnează informații despre model.

        Returns:
            Dict cu detalii model
        """

        return {
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'device': self.device,
            'batch_size': self.batch_size,
            'framework': 'sentence-transformers'
        }


def batch_generate_embeddings(
    texts: List[str],
    model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
    batch_size: int = 128,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Utility function: generează embeddings cu configurare automată.

    Simplu de folosit pentru task-uri quick.

    Args:
        texts: Liste de texte
        model_name: Model
        batch_size: Batch size
        use_gpu: Use GPU if available

    Returns:
        Numpy array (N, 768) unde N = len(texts)

    Exemplu:
        >>> texts = ["Text 1", "Text 2", "Text 3"]
        >>> embeddings = batch_generate_embeddings(texts)
        >>> print(embeddings.shape)
        "(3, 768)"
    """

    device = 'cuda' if use_gpu else 'cpu'
    generator = EmbeddingGenerator(
        model_name=model_name,
        device=device,
        batch_size=batch_size
    )

    embeddings = generator.generate_embeddings(texts, show_progress=True)

    # Convertire la numpy array
    return np.array(embeddings)


# Exemplu de utilizare
if __name__ == "__main__":
    # Test texts (Romanian + English)
    test_texts = [
        "Aria unui pătrat cu latura de 5 cm este 25 cm².",
        "Perimetrul pătratulu este 4 × latura.",
        "Într-un triunghi dreptunghic, teorema lui Pitagora spune că a² + b² = c².",
        "The square root of 16 is 4.",
        "Multiplication is the inverse operation of division.",
    ]

    # Inițializează generator
    generator = EmbeddingGenerator(device='cpu')

    # Generează embeddings
    embeddings = generator.generate_embeddings(test_texts, show_progress=False)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings[0].shape}")
    print(f"Model info: {generator.get_model_info()}")

    # Test similarity
    query = "Cum se calculează aria pătratulu?"
    query_emb = generator.generate_single_embedding(query)

    print(f"\nQuery: {query}")
    print("Similarities:")

    from scipy.spatial.distance import cosine

    for i, text in enumerate(test_texts):
        similarity = 1 - cosine(query_emb, embeddings[i])
        print(f"  {similarity:.2%} - {text[:50]}")
