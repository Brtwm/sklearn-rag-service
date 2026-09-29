import io
import json

import pytest

from app.scripts import load_corpus


PAGE_HTML = """
<nav>Site navigation should not enter the corpus</nav>
<article class="bd-article">
  <section id="linear-models">
    <h1>Linear models<a class="headerlink" href="#linear-models">#</a></h1>
    <p>Introduction to linear estimators.</p>
    <section id="ridge-regression">
      <h2>Ridge regression<a class="headerlink" href="#ridge-regression">#</a></h2>
      <p>Ridge applies an L2 penalty to coefficients.</p>
      <div class="highlight-python"><div class="highlight"><pre><span>model = Ridge()</span>
model.fit(X, y)</pre></div><button class="copybtn">Copy</button></div>
    </section>
    <section id="lasso">
      <h2>Lasso</h2>
      <p>Lasso applies an L1 penalty.</p>
    </section>
  </section>
</article>
"""
PAGE_URL = "https://scikit-learn.org/1.9/modules/linear_model.html"


def test_extract_sections_keeps_article_sections_and_code_without_duplicates() -> None:
    sections = load_corpus.extract_sections(PAGE_HTML, PAGE_URL)

    assert [doc.metadata["source"] for doc in sections] == [
        f"{PAGE_URL}#linear-models",
        f"{PAGE_URL}#ridge-regression",
        f"{PAGE_URL}#lasso",
    ]
    assert "Introduction to linear estimators." in sections[0].page_content
    assert "Ridge applies" not in sections[0].page_content
    assert "Linear models" in sections[1].page_content
    assert "model = Ridge()\nmodel.fit(X, y)" in sections[1].page_content
    assert "Lasso applies" not in sections[1].page_content
    assert all("Site navigation" not in doc.page_content for doc in sections)
    assert all("Copy" not in doc.page_content for doc in sections)
    assert all("#" not in doc.page_content for doc in sections)


def test_extract_sections_rejects_missing_article_or_section() -> None:
    with pytest.raises(ValueError, match="article"):
        load_corpus.extract_sections("<main><p>Empty</p></main>", PAGE_URL)
    with pytest.raises(ValueError, match="section"):
        load_corpus.extract_sections("<article class='bd-article'>Empty</article>", PAGE_URL)


def test_extract_sections_keeps_text_around_code_without_nested_section_text() -> None:
    html = (
        "<article class='bd-article'><section id='parent'><h1>Parent</h1>"
        "<div><p>Before code.</p><pre>run()</pre><p>After code.</p></div>"
        "<div><section id='child'><h2>Child</h2><p>Child text.</p></section></div>"
        "</section></article>"
    )

    parent, child = load_corpus.extract_sections(html, PAGE_URL)

    assert parent.page_content.index("Before code.") < parent.page_content.index("run()")
    assert parent.page_content.index("run()") < parent.page_content.index("After code.")
    assert "Child text." not in parent.page_content
    assert "Child text." in child.page_content


def test_extract_sections_keeps_article_topic_aside() -> None:
    html = (
        "<article class='bd-article'><section id='guidance'><h1>Guidance</h1>"
        "<aside class='topic'><p>Choose the estimator for the data.</p></aside>"
        "</section></article>"
    )

    [section] = load_corpus.extract_sections(html, PAGE_URL)

    assert "Choose the estimator for the data." in section.page_content


def test_chunk_documents_preserves_code_and_stable_ids(tmp_path) -> None:
    sections = load_corpus.extract_sections(PAGE_HTML, PAGE_URL)
    first = load_corpus.chunk_documents(sections)
    second = load_corpus.chunk_documents(sections)

    assert [doc.id for doc in first] == [doc.id for doc in second]
    assert len({doc.id for doc in first}) == len(first)
    assert all(doc.id for doc in first)
    assert any("model = Ridge()\nmodel.fit(X, y)" in doc.page_content for doc in first)
    assert all("Ridge applies" not in doc.page_content for doc in first if doc.metadata["section_id"] == "lasso")

    output = tmp_path / "chunks.jsonl"
    load_corpus.save_chunks(first, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [doc.id for doc in first]
    assert rows[1]["metadata"]["source"] == f"{PAGE_URL}#ridge-regression"


def test_long_code_block_is_not_split_between_chunks() -> None:
    code = "\n".join(f"value_{index} = {index}" for index in range(120))
    html = (
        "<article class='bd-article'><section id='example'>"
        "<h1>Example</h1><p>Before the example.</p>"
        f"<div class='highlight'><pre>{code}</pre></div>"
        "<p>After the example.</p></section></article>"
    )

    chunks = load_corpus.chunk_documents(load_corpus.extract_sections(html, PAGE_URL))

    assert len(chunks) >= 2
    assert sum(code in chunk.page_content for chunk in chunks) == 1
    assert all(chunk.metadata["source"] == f"{PAGE_URL}#example" for chunk in chunks)


def test_load_url_corpus_uses_ten_fixed_pages_in_order(monkeypatch) -> None:
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        return io.BytesIO(PAGE_HTML.encode("utf-8"))

    monkeypatch.setattr(load_corpus, "urlopen", fake_urlopen)
    docs = load_corpus.load_url_corpus()

    expected_pages = [
        "linear_model", "tree", "model_evaluation", "ensemble", "cross_validation",
        "preprocessing", "compose", "grid_search", "impute", "feature_selection",
    ]
    assert requested == [f"https://scikit-learn.org/1.9/modules/{name}.html" for name in expected_pages]
    assert [doc.metadata["page_url"] for doc in docs[::3]] == requested


def test_failed_download_does_not_save_partial_corpus(monkeypatch, tmp_path) -> None:
    def fake_urlopen(url, timeout):
        if url.endswith("tree.html"):
            raise OSError("network failure")
        return io.BytesIO(PAGE_HTML.encode("utf-8"))

    monkeypatch.setattr(load_corpus, "urlopen", fake_urlopen)
    monkeypatch.setattr(load_corpus, "SEED_URLS", [PAGE_URL, "https://scikit-learn.org/1.9/modules/tree.html"])
    output = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(load_corpus, "OUTPUT_PATH", output)

    with pytest.raises(OSError, match="network failure"):
        load_corpus.main()
    assert not output.exists()
