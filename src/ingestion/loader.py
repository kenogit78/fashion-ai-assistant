import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import pypdf


@dataclass
class Document:
    content: str
    source_file: str
    doc_type: str  # 'catalog', 'trend_report', 'campaign_brief'
    metadata: dict = field(default_factory=dict)


def load_csv(filepath: str) -> List[Document]:
    """Load product catalog CSV. Each row becomes one document."""
    docs = []
    df = pd.read_csv(filepath, on_bad_lines='skip')

    # H&M dataset columns — adjust if using different dataset
    text_columns = [col for col in df.columns
                    if df[col].dtype == object]

    for idx, row in df.iterrows():
        # Combine all text fields into one content string
        parts = []
        for col in text_columns:
            if pd.notna(row.get(col)):
                parts.append(f"{col}: {row[col]}")

        content = " | ".join(parts)

        if len(content.strip()) < 10:
            continue  # Skip empty rows

        docs.append(Document(
            content=content,
            source_file=Path(filepath).name,
            doc_type="catalog",
            metadata={"row_index": idx, "columns": text_columns}
        ))

        # Limit to 5000 rows for development speed
        if idx >= 4999:
            print(f"Loaded 5000 rows from {filepath}")
            break

    print(f"Loaded {len(docs)} documents from {filepath}")
    return docs


def load_pdf(filepath: str) -> List[Document]:
    """Load PDF — each page becomes one document."""
    docs = []
    reader = pypdf.PdfReader(filepath)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or len(text.strip()) < 50:
            continue  # Skip blank or near-blank pages

        docs.append(Document(
            content=text.strip(),
            source_file=Path(filepath).name,
            doc_type="trend_report",
            metadata={"page_number": page_num + 1,
                      "total_pages": len(reader.pages)}
        ))

    print(f"Loaded {len(docs)} pages from {filepath}")
    return docs


def load_text(filepath: str) -> List[Document]:
    """Load plain text — split on --- delimiter for campaign briefs."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    sections = [s.strip() for s in raw.split("---") if s.strip()]
    docs = []

    for idx, section in enumerate(sections):
        if len(section) < 20:
            continue

        docs.append(Document(
            content=section,
            source_file=Path(filepath).name,
            doc_type="campaign_brief",
            metadata={"brief_index": idx}
        ))

    print(f"Loaded {len(docs)} campaign briefs from {filepath}")
    return docs


def load_directory(data_dir: str) -> List[Document]:
    """Load all supported files from a directory."""
    all_docs = []
    data_path = Path(data_dir)

    for filepath in sorted(data_path.iterdir()):
        suffix = filepath.suffix.lower()

        if suffix == ".csv":
            all_docs.extend(load_csv(str(filepath)))
        elif suffix == ".pdf":
            all_docs.extend(load_pdf(str(filepath)))
        elif suffix == ".txt":
            all_docs.extend(load_text(str(filepath)))
        else:
            print(f"Skipping unsupported file: {filepath.name}")

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs