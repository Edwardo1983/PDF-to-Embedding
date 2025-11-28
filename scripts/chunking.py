"""
chunking.py - Split text în chunks inteligente cu deduplicare

Implementează:
1. Smart splitting (la puncuație, nu sparge cuvinte)
2. Overlap pentru context între chunks
3. Deduplicare MD5 (skip headers/footers repetate)
4. Filtrare chunk-uri prea mici

Un chunk bun e ~500 caractere - suficient pentru context semantic,
dar nu prea mare pentru embedding efficiency.
"""

import hashlib
import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """O bucată de text din document"""
    text: str  # Textul chunkului
    chunk_id: str  # ID unic generat din hash
    chunk_hash: str  # MD5 hash pentru deduplicare
    source_page: int  # Pagina din care provine (dacă disponibil)
    page_offset: int  # Offset în pagină
    metadata: Dict = None  # Metadate suplimentare


class TextChunker:
    """
    Chunker inteligent cu overlap și deduplicare.

    Utilizare:
        chunker = TextChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk_text("Text lung din PDF...")
        print(f"Generated {len(chunks)} chunks")
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_length: int = 50
    ):
        """
        Inițializează chunker.

        Args:
            chunk_size: Lungimea țintă a unui chunk (caractere)
            overlap: Caractere care se repetă între chunks (context)
            min_chunk_length: Chunk-uri mai mici sunt ignorate
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length

        logger.info(
            f"TextChunker initialized: "
            f"size={chunk_size}, overlap={overlap}, min={min_chunk_length}"
        )

    def chunk_text(
        self,
        text: str,
        source_page: int = None,
        remove_duplicates: bool = True
    ) -> List[Chunk]:
        """
        Splittează text în chunks inteligente.

        Strategie:
        1. Splittează la punctuație (. ! ?)
        2. Menține overlap pentru context
        3. Filtrează chunk-uri mici
        4. (Optional) Deduplica

        Args:
            text: Text complet din PDF/document
            source_page: Pagina sursă (metadata)
            remove_duplicates: Să elimini duplicate

        Returns:
            Lista de Chunk objects

        Exemplu:
            >>> chunker = TextChunker(chunk_size=500, overlap=50)
            >>> text = "Matem... (1000 chars)"
            >>> chunks = chunker.chunk_text(text, source_page=1)
            >>> print(f"Generated {len(chunks)} chunks")
            "Generated 3 chunks"
        """

        if not text or not text.strip():
            logger.warning("Empty text provided to chunk_text")
            return []

        # Curăță text
        text = self._clean_text(text)

        # Splittează în chunks
        raw_chunks = self._split_chunks(text)

        # Convertire în Chunk objects
        chunk_objects = []
        seen_hashes: Set[str] = set()

        for i, chunk_text in enumerate(raw_chunks):
            # Filtrează chunk-uri prea mici
            if len(chunk_text) < self.min_chunk_length:
                continue

            # Calculează hash pentru deduplicare
            chunk_hash = self._calculate_hash(chunk_text)

            # Sări peste duplicate
            if remove_duplicates and chunk_hash in seen_hashes:
                logger.debug(f"Skipping duplicate chunk {i}")
                continue

            seen_hashes.add(chunk_hash)

            # Creează Chunk object
            chunk_id = f"chunk_{len(chunk_objects)}"
            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                chunk_hash=chunk_hash,
                source_page=source_page or 0,
                page_offset=i,
                metadata={
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'has_numbers': any(c.isdigit() for c in chunk_text),
                    'has_equations': '=' in chunk_text or '≈' in chunk_text
                }
            )

            chunk_objects.append(chunk)

        logger.info(
            f"Chunked text: {len(raw_chunks)} raw → {len(chunk_objects)} deduplicated chunks"
        )

        return chunk_objects

    def _split_chunks(self, text: str) -> List[str]:
        """
        Splittează text în chunks cu overlap.

        Strategie:
        1. Splittează la puncte/pauze
        2. Recombine în aproximativ chunk_size
        3. Adaugă overlap

        Args:
            text: Text complet

        Returns:
            Lista de text chunks (string-uri)
        """

        # Splittează la puncte și pauze logice
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Adaugă o propoziție la chunk curent
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            # Dacă chunk devine prea mare, salvează și start nou
            if len(test_chunk) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)

                # Startează cu overlap din proposiția curentă
                current_chunk = sentence

            else:
                current_chunk = test_chunk

        # Adaugă chunk-ul final
        if current_chunk:
            chunks.append(current_chunk)

        # Adaugă overlap între chunks (replicare cuvinte)
        if self.overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Adaugă overlap între chunks pentru context.

        Exemplu:
            chunk1: "Aria pătrat = latura² ... [final 50 chars]"
            chunk2: "[ultima 50 chars din chunk1] ... Exercițiu:"

        Args:
            chunks: Lista de chunks fără overlap

        Returns:
            Lista de chunks cu overlap
        """

        if len(chunks) <= 1:
            return chunks

        overlapped = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # Primul chunk rămâne neschimbat
                overlapped.append(chunk)
            else:
                # Adaugă overlappul din chunk-ul anterior
                prev_overlap = overlapped[i - 1][-self.overlap:]
                new_chunk = prev_overlap + " " + chunk
                overlapped.append(new_chunk)

        return overlapped

    def _split_sentences(self, text: str) -> List[str]:
        """
        Splittează text în propoziții logice.

        Delimitatori: . ! ? \n

        Args:
            text: Text complet

        Returns:
            Lista de propoziții
        """

        import re

        # Splittează la puncuație
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Splittează și la line breaks
        expanded = []
        for sent in sentences:
            parts = sent.split('\n')
            expanded.extend([p.strip() for p in parts if p.strip()])

        return expanded

    def _clean_text(self, text: str) -> str:
        """
        Curăță text de caractere nefolositoare.

        Args:
            text: Text brut din PDF

        Returns:
            Text curat
        """

        # Înlocuiește whitespace-uri multiple
        import re
        text = re.sub(r'\s+', ' ', text)

        # Elimină anumite caractere de control
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        # Trim whitespace-uri
        text = text.strip()

        return text

    def _calculate_hash(self, text: str) -> str:
        """
        Calculează MD5 hash pentru text.

        Folosit pentru deduplicare.

        Args:
            text: Text pentru hash

        Returns:
            Hex MD5 hash
        """

        return hashlib.md5(text.encode()).hexdigest()


def chunk_multiple_texts(
    texts: List[Tuple[str, int]],  # (text, page_number)
    chunk_size: int = 500,
    overlap: int = 50,
    remove_duplicates: bool = True
) -> List[Chunk]:
    """
    Procesează mai multe texte (din diferite pagini/fișiere).

    Utility pentru procesare batch.

    Args:
        texts: Listă de tuple (text, page_number)
        chunk_size: Mărime chunk
        overlap: Overlap
        remove_duplicates: Să deduplica

    Returns:
        Lista totală de chunks din toate textele

    Exemplu:
        >>> texts = [
        ...     ("Pagina 1 text...", 1),
        ...     ("Pagina 2 text...", 2),
        ... ]
        >>> all_chunks = chunk_multiple_texts(texts)
        >>> print(f"Total chunks: {len(all_chunks)}")
    """

    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    all_chunks = []

    for text, page_num in texts:
        chunks = chunker.chunk_text(text, source_page=page_num, remove_duplicates=remove_duplicates)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks from {len(texts)} texts: {len(all_chunks)}")
    return all_chunks


# Exemplu de utilizare
if __name__ == "__main__":
    # Test text
    test_text = """
    Capitolul 1: Numere naturale. Numerele naturale sunt 0, 1, 2, 3, ...
    Folosim aceste numere pentru a număra obiectele.

    Exemplu: Am 5 mere și 3 portocale. În total am 8 fructe.

    Exercițiu 1: Calculează 2 + 3.
    Exercițiu 2: Calculează 5 + 4.

    Răspunsuri: 2 + 3 = 5. 5 + 4 = 9.

    Capitolul 2: Adunarea numerelor. Adunarea este operația de combinare a două numere.

    Termeni: În operația 3 + 4 = 7, 3 și 4 sunt termenii, iar 7 este suma.
    """ * 10  # Repetă pentru text mai lung

    # Chunk text
    chunker = TextChunker(chunk_size=200, overlap=30)
    chunks = chunker.chunk_text(test_text, source_page=1, remove_duplicates=True)

    print(f"Generated {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks[:3]):  # Arată primele 3
        print(f"Chunk {i}:")
        print(f"  Text: {chunk.text[:80]}...")
        print(f"  Hash: {chunk.chunk_hash}")
        print(f"  Metadata: {chunk.metadata}\n")
