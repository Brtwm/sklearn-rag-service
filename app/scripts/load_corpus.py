"""Загрузка разделов документации scikit-learn и локальных .md-файлов в JSONL."""

import json
from pathlib import Path
from urllib.request import urlopen
from uuid import NAMESPACE_URL, uuid5

from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEED_URLS = [
    "https://scikit-learn.org/1.9/modules/linear_model.html",
    "https://scikit-learn.org/1.9/modules/tree.html",
    "https://scikit-learn.org/1.9/modules/model_evaluation.html",
    "https://scikit-learn.org/1.9/modules/ensemble.html",
    "https://scikit-learn.org/1.9/modules/cross_validation.html",
    "https://scikit-learn.org/1.9/modules/preprocessing.html",
    "https://scikit-learn.org/1.9/modules/compose.html",
    "https://scikit-learn.org/1.9/modules/grid_search.html",
    "https://scikit-learn.org/1.9/modules/impute.html",
    "https://scikit-learn.org/1.9/modules/feature_selection.html",
]

LOCAL_DIR = Path("data/local")
OUTPUT_PATH = Path("data/corpus_chunks.jsonl")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _content_blocks(element: Tag) -> list[str]:
    if element.name == "section":
        return []
    if element.name == "pre":
        code = element.get_text("", strip=False).strip()
        return [f"```\n{code}\n```"] if code else []
    if element.find(["pre", "section"]) is None:
        text = element.get_text(" ", strip=True)
        return [text] if text else []
    blocks = []
    for child in element.children:
        if isinstance(child, Tag):
            blocks.extend(_content_blocks(child))
        elif child.strip():
            blocks.append(child.strip())
    return blocks


def extract_sections(html: str, url: str) -> list[Document]:
    """Return each article section with its own text and an anchored source URL."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("article.bd-article")
    if article is None:
        raise ValueError(f"Missing article in {url}")
    for tag in article.select("script, style, nav, footer, button, a.headerlink"):
        tag.decompose()
    sections = article.select("section[id]")
    if not sections:
        raise ValueError(f"Missing section in {url}")

    docs: list[Document] = []
    for section in sections:
        headings = []
        current: Tag | None = section
        while current is not None and current is not article:
            if current.name == "section":
                heading = current.find(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False)
                if heading is not None:
                    headings.append(heading.get_text(" ", strip=True))
            current = current.parent if isinstance(current.parent, Tag) else None
        headings.reverse()
        if not headings:
            raise ValueError(f"Missing section heading in {url}#{section['id']}")

        blocks = []
        for child in section.children:
            if not isinstance(child, Tag) or child.name == "section" or child.name in {
                "h1", "h2", "h3", "h4", "h5", "h6",
            }:
                continue
            blocks.extend(_content_blocks(child))
        if not blocks:
            continue
        docs.append(Document(
            page_content="\n\n".join([*headings, *blocks]),
            metadata={
                "source": f"{url}#{section['id']}",
                "page_url": url,
                "section_id": section["id"],
                "title": headings[-1],
                "_headings": headings,
                "_blocks": blocks,
            },
        ))
    if not docs:
        raise ValueError(f"No article content in {url}")
    return docs


def load_url_corpus() -> list[Document]:
    """Download the ten selected documentation pages, in the listed order."""
    docs: list[Document] = []
    for url in SEED_URLS:
        print(f"Loading {url} ...")
        with urlopen(url, timeout=30) as response:
            html = response.read().decode("utf-8")
        page_docs = extract_sections(html, url)
        docs.extend(page_docs)
        print(f"  -> {len(page_docs)} sections")
    return docs


def load_local_corpus() -> list[Document]:
    """Подтянуть любые `*.md` из `data/local/` - внутренние документы сервиса."""
    if not LOCAL_DIR.exists():
        return []
    docs: list[Document] = []
    for path in sorted(LOCAL_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"local://{path.name}",
                    "title": path.stem.replace("_", " ").title(),
                }
            )
        )
        print(f"Local file: {path.name}, ({len(text)} chars)")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split within sections, keep code blocks intact, and assign stable IDs."""
    chunks: list[Document] = []
    for doc in docs:
        metadata = {key: value for key, value in doc.metadata.items() if not key.startswith("_")}
        headings = doc.metadata.get("_headings")
        if headings is None:
            parts = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            ).split_text(doc.page_content)
        else:
            prefix = "\n".join(headings)
            size = max(100, CHUNK_SIZE - len(prefix) - 2)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=min(CHUNK_OVERLAP, size // 4),
            )
            blocks = []
            for block in doc.metadata["_blocks"]:
                blocks.extend([block] if block.startswith("```\n") else splitter.split_text(block))
            parts = []
            current: list[str] = []
            for block in blocks:
                joined = "\n\n".join([*current, block])
                if current and len(joined) > size:
                    parts.append(f"{prefix}\n\n" + "\n\n".join(current))
                    overlap: list[str] = []
                    for previous in reversed(current):
                        if len("\n\n".join([previous, *overlap])) > CHUNK_OVERLAP:
                            break
                        overlap.insert(0, previous)
                    current = overlap if len("\n\n".join([*overlap, block])) <= size else []
                current.append(block)
            if current:
                parts.append(f"{prefix}\n\n" + "\n\n".join(current))

        for index, content in enumerate(parts):
            chunks.append(Document(
                id=str(uuid5(NAMESPACE_URL, f"{metadata['source']}:{index}:{content}")),
                page_content=content,
                metadata=metadata,
            ))
    return chunks


def save_chunks(chunks: list[Document], path: Path) -> None:
    """Write chunks as JSONL with stable IDs and source metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            row = {"id": chunk.id, "content": chunk.page_content, "metadata": dict(chunk.metadata)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {path}")


def main() -> None:
    """Download, split and save the corpus only after all pages load."""
    url_docs = load_url_corpus()
    local_docs = load_local_corpus()
    chunks = chunk_documents(url_docs + local_docs)
    print(f"\nTotal: {len(url_docs) + len(local_docs)} sections/docs -> {len(chunks)} chunks")
    save_chunks(chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()
