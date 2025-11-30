"""
Builds a Jupyter notebook that mirrors the structured, numbered style of kaggle_notebook.ipynb,
but sources code from the Python modules in `scripts/`.

Features:
- UTF-8 reads/writes; skips missing files gracefully (warns to stderr).
- Numbered, explained cells (markdown + code), no absolute filesystem paths.
- Kernelspec metadata for portability (e.g., Kaggle).
"""

import argparse
import sys
from pathlib import Path

from nbformat import v4 as nbf

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_MODULES = [
    # Order follows pipeline flow: extract -> ocr -> chunk -> embed -> upload
    Path("scripts/pdf_extractor.py"),
    Path("scripts/ocr_processor.py"),
    Path("scripts/chunking.py"),
    Path("scripts/embedding_generator.py"),
    Path("scripts/supabase_uploader.py"),
]

FRIENDLY_TITLES = {
    "chunking.py": "Chunking Utilities",
    "embedding_generator.py": "Embedding Generator",
    "ocr_processor.py": "OCR Processor",
    "pdf_extractor.py": "PDF Extractor",
    "supabase_uploader.py": "Supabase Uploader",
}

FRIENDLY_DESCRIPTIONS = {
    "chunking.py": [
        "- Split text into chunks with overlap.",
        "- Deduplicate by hash and drop too-short chunks.",
    ],
    "embedding_generator.py": [
        "- Generate sentence-transformer embeddings in batches.",
        "- Include helpers for metadata and single-query embeddings.",
    ],
    "ocr_processor.py": [
        "- PaddleOCR wrapper with selective processing and confidence filters.",
        "- Handles batch processing and priority thresholds.",
    ],
    "pdf_extractor.py": [
        "- Extract text/images via PyMuPDF with image priority heuristics.",
        "- Supports optional page limits for quick debug.",
    ],
    "supabase_uploader.py": [
        "- Batch upload/query utilities for Supabase pgvector with retries.",
        "- Simple verify and similarity query helpers.",
    ],
}


def friendly_name(module_path: Path) -> str:
    return FRIENDLY_TITLES.get(module_path.name, module_path.stem.replace("_", " ").title())


def build_notebook(modules, output_path: Path) -> None:
    nb = nbf.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    cells = []

    # Intro section similar to kaggle_notebook.ipynb
    cells.append(nbf.new_markdown_cell("# PDF-to-Embedding Pipeline"))
    cells.append(
        nbf.new_markdown_cell(
            "## Cell 1: Setup & Dependencies\n"
            "- Instaleaza/importeaza dependintele necesare (daca rulezi pe Kaggle, multe sunt preinstalate).\n"
            "- Configureaza mediul si logger-ul dupa nevoie."
        )
    )
    cells.append(
        nbf.new_code_cell(
            "# Setup & dependencies\n"
            "import logging\n"
            "logging.basicConfig(level=logging.INFO)\n"
            "print(\"\\nInstalling dependencies...\")\n"
            "!pip install -q \\\n"
            "    PyMuPDF>=1.23.0 \\\n"
            "    paddleocr>=2.7.0 \\\n"
            "    sentence-transformers>=2.2.2 \\\n"
            "    supabase>=2.0.0 \\\n"
            "    numpy>=1.24.0 \\\n"
            "    pandas>=2.0.0 \\\n"
            "    scipy>=1.10.0 \\\n"
            "    tqdm>=4.65.0 \\\n"
            "    pyyaml>=6.0\n"
            "print(\"\\n✅ Dependencies installed successfully\")\n"
        )
    )

    # Align with kaggle_notebook flow: imports/config as Cell 2.
    cells.append(
        nbf.new_markdown_cell(
            "## Cell 2: Import Modules & Configuration\n"
            "- Optional: importa modulele sau seteaza variabile de configurare.\n"
            "- Codul complet al modulelor urmeaza in celulele dedicate."
        )
    )
    cells.append(
        nbf.new_code_cell(
            "# Import modules (if packaged); below cells contain full source for reference.\n"
            "# from scripts import chunking, embedding_generator, ocr_processor, pdf_extractor, supabase_uploader\n"
        )
    )

    cell_number = 3

    for module in modules:
        module_path = (REPO_ROOT / module).resolve()
        if not module_path.exists():
            print(f"[WARN] Missing module: {module_path}", file=sys.stderr)
            continue

        code = module_path.read_text(encoding="utf-8")
        title = friendly_name(module_path)
        desc = FRIENDLY_DESCRIPTIONS.get(module_path.name)
        extra = "" if not desc else "\n" + "\n".join(desc)
        cells.append(nbf.new_markdown_cell(f"## Cell {cell_number}: {title}{extra}"))
        cells.append(nbf.new_code_cell(code))
        cell_number += 1

    # Closing cells similar to kaggle_notebook
    cells.append(nbf.new_markdown_cell("## DONE! 🎉"))
    cells.append(
        nbf.new_code_cell(
            "print('\\n' + '='*70)\nprint('Notebook generation complete.')\n"
        )
    )

    nb.cells = cells
    output_path = output_path.resolve()
    output_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"[OK] Notebook written to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate NOTEBOOK.ipynb from Python modules."
    )
    parser.add_argument(
        "--output",
        default="NOTEBOOK.ipynb",
        help="Output notebook path (default: NOTEBOOK.ipynb)",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        type=Path,
        default=DEFAULT_MODULES,
        help="List of module paths to include (default: scripts/*.py)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_notebook(args.modules, Path(args.output))
