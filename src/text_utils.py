from pathlib import Path
from pypdf import PdfReader
from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents():
    """读取 data 目录下的 md/txt/pdf；每份文件保留来源。"""
    docs = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".md", ".txt"}:
                text = path.read_text(encoding="utf-8", errors="replace")
            elif suffix == ".pdf":
                reader = PdfReader(str(path))
                pages = []
                for page_no, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(f"[PDF 第 {page_no} 页]\n{page_text}")
                text = "\n\n".join(pages)
            else:
                continue
        except Exception as exc:
            print(f"[跳过文件] {path}: {exc}")
            continue

        if text.strip():
            docs.append({"source": str(path.relative_to(DATA_DIR)), "text": text})

    return docs


def split_text(text, chunk_size=None, overlap=None):
    """字符级教学切分：优先在段落/换行/句号附近切分。"""
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = CHUNK_OVERLAP if overlap is None else overlap
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("要求 chunk_size > overlap >= 0")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            floor = start + int(chunk_size * 0.65)
            candidates = [
                text.rfind("\n\n", floor, end),
                text.rfind("\n", floor, end),
                text.rfind("。", floor, end),
                text.rfind(". ", floor, end),
                text.rfind("；", floor, end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks
