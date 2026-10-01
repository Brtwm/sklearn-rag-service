from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.main import app, respond
from app.rag import chain, reranker, retrieval
from tests.test_index_corpus import SmallEmbeddings
from tests.test_retrieval import hit


@pytest.fixture(autouse=True)
def reset_cached_model(monkeypatch):
    monkeypatch.setattr(reranker, "_reranker", None)


def docs():
    return [Document(id=str(i), page_content=f"Passage {i}", metadata={
        "source": f"https://example.org/page#{i}", "title": f"Section {i}",
    }) for i in range(6)]


def test_model_loads_once_even_with_concurrent_initialization() -> None:
    with patch("app.rag.reranker.CrossEncoder") as encoder:
        with ThreadPoolExecutor(max_workers=4) as pool:
            instances = list(pool.map(lambda _: reranker.get_reranker(), range(8)))
    assert all(instance is instances[0] for instance in instances)
    encoder.assert_called_once_with(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu", max_length=512,
    )


def test_rerank_scores_raw_question_and_text_preserving_ties_and_metadata() -> None:
    with patch("app.rag.reranker.CrossEncoder") as encoder:
        encoder.return_value.predict.return_value = [0.1, 0.8, 0.8, -0.2, 0.9, 0.3]
        ranker = reranker.get_reranker()
        original = docs()
        results = ranker.rerank("Как работает Ridge?", original)
    encoder.return_value.predict.assert_called_once_with(
        [("Как работает Ridge?", doc.page_content) for doc in original],
        batch_size=8, show_progress_bar=False,
    )
    assert [doc.id for doc, _ in results] == ["4", "1", "2", "5", "0", "3"]
    assert results[0] == (original[4], 0.9)
    assert [doc.id for doc in original] == [str(i) for i in range(6)]


@pytest.mark.parametrize("n", [0, 1, 3])
def test_rerank_empty_and_short_candidate_lists(n: int) -> None:
    with patch("app.rag.reranker.CrossEncoder") as encoder:
        encoder.return_value.predict.return_value = list(range(n))
        ranker = reranker.get_reranker()
        result = ranker.rerank("Ridge?", docs()[:n])
    assert len(result) == n
    if n == 0:
        encoder.return_value.predict.assert_not_called()


def test_hybrid_rerank_ranks_all_unique_candidates_before_cutting_top_four() -> None:
    client = Mock()
    client.query_points.return_value.points = [hit(str(i)) for i in range(6)] + [hit("1")]
    with patch("app.rag.reranker.CrossEncoder") as encoder:
        encoder.return_value.predict.return_value = [0.1, 0.8, 0.8, -0.2, 0.9, 0.3]
        search = retrieval.CorpusRetriever(
            client, "physical", SmallEmbeddings(), "hybrid_rerank", 4,
            reranker=reranker.get_reranker(),
        )
        result = search.search_with_score("Ridge?", limit=4)
    assert [doc.id for doc, _ in result] == ["4", "1", "2", "5"]
    assert client.query_points.call_args.kwargs["limit"] == 40
    assert len(encoder.return_value.predict.call_args.args[0]) == 6
    client.query_points.assert_called_once()


def test_reranked_invoke_returns_top_four_in_score_order() -> None:
    client = Mock()
    client.query_points.return_value.points = [hit(str(i)) for i in range(6)]
    with patch("app.rag.reranker.CrossEncoder") as encoder:
        encoder.return_value.predict.return_value = list(range(6))
        search = retrieval.CorpusRetriever(
            client, "physical", SmallEmbeddings(), "hybrid_rerank", 4,
            reranker=reranker.get_reranker(),
        )
        assert [doc.id for doc in search.invoke("Ridge?")] == ["5", "4", "3", "2"]


@pytest.fixture
def rag_with_reranker(monkeypatch):
    monkeypatch.setattr(chain.settings, "retrieval_mode", "hybrid_rerank")
    client = Mock()
    client.collection_exists.return_value = True
    client.query_points.return_value.points = [hit(str(i)) for i in range(6)]
    with patch("app.rag.chain.QdrantClient", return_value=client), patch(
        "app.rag.chain.verify_index_schema"
    ), patch("app.rag.chain.get_embeddings", return_value=SmallEmbeddings()), patch(
        "app.rag.reranker.CrossEncoder"
    ) as encoder, patch("app.rag.chain.get_llm") as llm:
        encoder.return_value.predict.return_value = list(range(6))
        contexts = []

        def answer(prompt):
            contexts.append(prompt.to_messages()[-1].content)
            return "See [1] and [4]."

        llm.return_value = RunnableLambda(answer)
        yield client, encoder, llm, contexts


def test_rest_and_stream_citations_follow_reranked_order(rag_with_reranker) -> None:
    client, encoder, _, contexts = rag_with_reranker
    with TestClient(app) as http:
        response = http.post("/chat", json={"question": "Ridge?"})
        assert response.status_code == 200
        assert [source["url"] for source in response.json()["sources"]] == [
            f"https://example.org/page#{i}" for i in (5, 4, 3, 2)
        ]
        for rank, i in enumerate((5, 4, 3, 2), 1):
            assert f"[{rank}] Source: https://example.org/page#{i}" in contexts[-1]
        stream = respond("Как работает Ridge?", [])
        first = next(stream)
        assert first[0][-1]["content"] == ""
        events = list(stream)
        assert events[-1][0][-1]["content"] == "See [1] and [4]."
        for rank, i in enumerate((5, 4, 3, 2), 1):
            assert f"**[{rank}]** [Section](<https://example.org/page#{i}>)" in events[-1][3]
    assert client.query_points.call_count == 2
    assert encoder.return_value.predict.call_count == 2
    assert encoder.call_count == 1


def test_model_is_reused_after_qdrant_recovers(rag_with_reranker) -> None:
    _, encoder, _, _ = rag_with_reranker
    with patch("app.main.index_available") as available:
        available.return_value = True
        with TestClient(app) as http:
            assert http.get("/ready").status_code == 200
            available.return_value = False
            assert http.get("/ready").status_code == 503
            available.return_value = True
            assert http.get("/ready").status_code == 200
    encoder.assert_called_once()


def test_model_load_error_keeps_health_and_marks_rag_not_ready(rag_with_reranker) -> None:
    _, encoder, llm, _ = rag_with_reranker
    encoder.side_effect = OSError("Model unavailable")
    with TestClient(app) as http:
        assert http.get("/health").status_code == 200
        assert http.get("/ready").status_code == 503
        assert http.post("/chat", json={"question": "Ridge?"}).status_code == 503
    llm.assert_not_called()


def test_model_inference_error_returns_503_and_does_not_call_llm(rag_with_reranker) -> None:
    client, encoder, _, contexts = rag_with_reranker
    encoder.return_value.predict.side_effect = RuntimeError("Inference failed")
    with TestClient(app) as http:
        response = http.post("/chat", json={"question": "Ridge?"})
        assert response.status_code == 503
        assert "Retrieval temporarily unavailable" in response.json()["detail"]
        events = list(respond("Ridge?", []))
        assert "Поиск сейчас недоступен" in events[-1][0][-1]["content"]
    assert client.query_points.call_count == 2
    assert encoder.return_value.predict.call_count == 2
    assert contexts == []


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_other_modes_do_not_load_reranker(mode, monkeypatch) -> None:
    monkeypatch.setattr(chain.settings, "retrieval_mode", mode)
    with patch("app.rag.chain.QdrantClient"), patch("app.rag.chain.verify_index_schema"), patch(
        "app.rag.chain.get_embeddings", return_value=SmallEmbeddings()
    ), patch("app.rag.reranker.CrossEncoder") as encoder:
        search = chain.get_retriever()
        assert search.mode == mode
    encoder.assert_not_called()
