from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def test_gradio_page_loads_from_mounted_app() -> None:
    with patch("app.main.build_rag_chain", return_value=(MagicMock(), MagicMock())):
        with TestClient(app) as client:
            response = client.get("/")

    assert response.status_code == 200
    assert "scikit-learn docs RAG" in response.text
