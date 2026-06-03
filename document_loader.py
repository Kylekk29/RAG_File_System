import logging
import os
from typing import Callable, Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

LoaderFunc = Callable[[str], List[Document]]


def _load_text(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    return TextLoader(file_path, encoding="utf-8").load()


def _load_pdf(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(file_path).load()


def _load_csv(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(file_path, encoding="utf-8").load()


def _load_json(file_path: str) -> List[Document]:
    import json as json_mod
    with open(file_path, "r", encoding="utf-8") as f:
        data = json_mod.load(f)
    text = json_mod.dumps(data, ensure_ascii=False, indent=2)
    return [Document(page_content=text, metadata={"source": file_path})]


def _load_html(file_path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import BSHTMLLoader
        return BSHTMLLoader(file_path, open_encoding="utf-8").load()
    except ImportError:
        logger.warning("BSHTMLLoader unavailable (install beautifulsoup4+lxml). Falling back to TextLoader.")
        return _load_text(file_path)


def _load_docx(file_path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path).load()
    except ImportError:
        logger.warning("Docx2txtLoader unavailable (install python-docx). Falling back to raw text.")
        return _load_text(file_path)


def _load_pptx(file_path: str) -> List[Document]:
    try:
        from pptx import Presentation
        parts = []
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides, 1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            texts.append(t)
            if texts:
                parts.append(Document(
                    page_content="\n".join(texts),
                    metadata={"source": file_path, "page": i, "slide": i}
                ))
        if not parts:
            parts.append(Document(page_content="", metadata={"source": file_path}))
        return parts
    except ImportError:
        logger.warning("python-pptx unavailable (install python-pptx). Falling back to raw text.")
        return _load_text(file_path)


def _load_epub(file_path: str) -> List[Document]:
    try:
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(file_path)
        docs = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator="\n").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "chapter": item.get_name()}
                    ))
        if not docs:
            docs.append(Document(page_content="", metadata={"source": file_path}))
        return docs
    except ImportError:
        logger.warning(
            "EPUB support requires 'ebooklib' and 'beautifulsoup4'. "
            "Install with: pip install ebooklib beautifulsoup4"
        )
        raise ImportError(
            "EPUB support requires 'ebooklib' and 'beautifulsoup4'. "
            "Install with: pip install ebooklib beautifulsoup4"
        )


def _load_rtf(file_path: str) -> List[Document]:
    try:
        from striprtf.striprtf import rtf_to_text
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        text = rtf_to_text(raw)
        return [Document(page_content=text, metadata={"source": file_path})]
    except ImportError:
        logger.warning("striprtf unavailable (install striprtf). Falling back to TextLoader.")
        return _load_text(file_path)


LOADER_REGISTRY: Dict[str, LoaderFunc] = {
    ".txt": _load_text,
    ".md": _load_text,
    ".rst": _load_text,
    ".text": _load_text,
    ".log": _load_text,
    ".py": _load_text,
    ".js": _load_text,
    ".ts": _load_text,
    ".jsx": _load_text,
    ".tsx": _load_text,
    ".css": _load_text,
    ".scss": _load_text,
    ".less": _load_text,
    ".sql": _load_text,
    ".sh": _load_text,
    ".bat": _load_text,
    ".ps1": _load_text,
    ".yaml": _load_text,
    ".yml": _load_text,
    ".toml": _load_text,
    ".ini": _load_text,
    ".cfg": _load_text,
    ".conf": _load_text,
    ".xml": _load_text,
    ".svg": _load_text,
    ".tex": _load_text,
    ".pdf": _load_pdf,
    ".csv": _load_csv,
    ".tsv": _load_csv,
    ".json": _load_json,
    ".html": _load_html,
    ".htm": _load_html,
    ".xhtml": _load_html,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".epub": _load_epub,
    ".rtf": _load_rtf,
}

TEXT_LIKE_EXTENSIONS = frozenset({
    ".txt", ".md", ".rst", ".text", ".log",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".css", ".scss", ".less",
    ".sql", ".sh", ".bat", ".ps1",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".svg", ".tex",
    ".json", ".csv", ".tsv",
    ".html", ".htm", ".xhtml",
    ".rtf",
})

FILE_TYPE_LABELS: Dict[str, str] = {
    ".pdf": "PDF",
    ".txt": "TXT",
    ".md": "MD",
    ".rst": "RST",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".json": "JSON",
    ".html": "HTML",
    ".htm": "HTML",
    ".xhtml": "XHTML",
    ".docx": "DOCX",
    ".pptx": "PPTX",
    ".epub": "EPUB",
    ".rtf": "RTF",
    ".py": "PY",
    ".js": "JS",
    ".ts": "TS",
    ".jsx": "JSX",
    ".tsx": "TSX",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sql": "SQL",
    ".sh": "SH",
    ".bat": "BAT",
    ".ps1": "PS1",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
    ".tex": "TEX",
    ".log": "LOG",
    ".svg": "SVG",
    ".ini": "INI",
    ".cfg": "CFG",
}

FILE_TYPE_ICONS: Dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".csv": "spreadsheet",
    ".tsv": "spreadsheet",
    ".json": "code",
    ".html": "web",
    ".htm": "web",
    ".xhtml": "web",
    ".docx": "document",
    ".pptx": "slides",
    ".epub": "ebook",
    ".rtf": "document",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".jsx": "code",
    ".tsx": "code",
    ".css": "code",
    ".sql": "code",
    ".sh": "terminal",
    ".yaml": "config",
    ".yml": "config",
    ".toml": "config",
    ".xml": "code",
    ".log": "text",
    ".tex": "document",
    ".svg": "image",
}


def load_document(file_path: str) -> List[Document]:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    loader = LOADER_REGISTRY.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(LOADER_REGISTRY.keys()))}"
        )
    logger.info("Loading document: %s (type=%s)", os.path.basename(file_path), ext)
    docs = loader(file_path)
    logger.info("Loaded %d document(s) from %s", len(docs), os.path.basename(file_path))
    return docs


def is_text_like(file_path: str) -> bool:
    _, ext = os.path.splitext(file_path)
    return ext.lower() in TEXT_LIKE_EXTENSIONS


def get_supported_extensions() -> List[str]:
    return sorted(LOADER_REGISTRY.keys())


def get_file_type_label(ext: str) -> str:
    ext = ext.lower()
    return FILE_TYPE_LABELS.get(ext, ext.lstrip(".").upper())


def get_file_type_icon(ext: str) -> str:
    ext = ext.lower()
    return FILE_TYPE_ICONS.get(ext, "text")
