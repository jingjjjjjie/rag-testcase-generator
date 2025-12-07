import re
from pathlib import Path

PROJECT_ROOT = PROJECT_ROOT = Path(__file__).parent.parent

def split_by_questions(text):
    """
    Split into blocks using Q markers like "1)", "2)", ..., "10)"
    """
    pattern = r"(?m)^(?:\d{1,2}\))"
    matches = list(re.finditer(pattern, text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        chunks.append(chunk)

    return chunks