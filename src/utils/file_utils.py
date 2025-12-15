from pathlib import Path
import json
import os
import PyPDF2
from pathlib import Path

def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_json(data, path, verbose=False):
    """Save data to JSON file."""
    dir_path = os.path.dirname(path)
    if dir_path:  # Only create directory if path includes one
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Saved {len(data)} items to {path}")

def load_json(path,verbose=False):
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if verbose:
        print(f"Loaded {len(data)} items from {path}")
    return data

def extract_text_from_pdf(pdf_source) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_source: Either a file path (str/Path) or a file-like object (e.g., from Streamlit upload)

    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        # Check if it's a file path
        if isinstance(pdf_source, (str, Path)):
            pdf_path = Path(pdf_source)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        else:
            # Assume it's a file-like object (BytesIO, uploaded file, etc.)
            pdf_reader = PyPDF2.PdfReader(pdf_source)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}") from e
    return text