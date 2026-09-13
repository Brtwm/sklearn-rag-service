"""Загрузка корпуса: документация scikit-learn + локальные .md-файлы.

Делает четыре вещи подряд: тянет HTML-страницы по списку URL, читает
локальные `*.md` из `data/local/`, чистит HTML от script/nav/footer и
режет результат на чанки фиксированного размера. Финальный результат
- `data/corpus_chunks.jsonl`, который дальше едет в `index_corpus.py`.
"""
import json
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEED_URLS = [
    "https://scikit-learn.org/stable/modules/linear_model.html",
    "https://scikit-learn.org/stable/modules/tree.html",
    "https://scikit-learn.org/stable/modules/model_evaluation.html",
]

LOCAL_DIR = Path("data/local")
OUTPUT_PATH = Path("data/corpus_chunks.jsonl")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def clean_html(html: str) -> str:
    """Выкинуть script/style/nav/header/footer и вернуть чистый текст страницы."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_url_corpus() -> list[Document]:
    """Скачать страницы документации scikit-learn по списку SEED_URLS"""
    all_docs: list[Document] = []
    for url in SEED_URLS:
        print(f"Loading {url} ...")
        loader = RecursiveUrlLoader(url=url, max_depth=1, extractor=clean_html)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  -> {len(docs)} pages")
    return all_docs


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
    """Нарезать документы на куски CHUNK_SIZE символов с overlap'ом"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def save_chunks(chunks: list[Document], path: Path) -> None:
    """Сохранить чанки в jsonl: каждая строка - {"content: ..., "metadata": ...}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            row = {"content": chunk.page_content, "metadata": dict(chunk.metadata)}
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"Saved {len(chunks)} chunks to {path}")


def main() -> None:
    """Точка входа: грузим, режем, сохраняем"""
    url_docs = load_url_corpus()
    local_docs = load_local_corpus()
    chunks = chunk_documents(url_docs + local_docs)
    print(f"\nTotal: {len(url_docs) + len(local_docs)} pages -> {len(chunks)} chunks")
    save_chunks(chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()