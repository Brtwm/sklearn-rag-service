import json
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI

from app.config import Settings
from app import llm as provider


def test_neuraldeep_settings_are_forwarded_to_chat_api():
    settings = Settings(_env_file=None, llm_api_key="test-only",
                        llm_base_url="https://api.neuraldeep.ru/v1",
                        llm_model="qwen3.6-35b-a3b-noreason")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": settings.llm_model, "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "L2 [1]"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        def factory(**kwargs):
            return ChatOpenAI(**kwargs, http_client=client)

        with patch.object(provider, "settings", settings), patch.object(provider, "ChatOpenAI", side_effect=factory):
            assert provider.get_llm().invoke("Ridge?").content == "L2 [1]"
    assert str(requests[0].url) == "https://api.neuraldeep.ru/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "qwen3.6-35b-a3b-noreason"


def test_neuraldeep_streaming_uses_standard_sse():
    settings = Settings(_env_file=None, llm_api_key="test-only",
                        llm_base_url="https://api.neuraldeep.ru/v1",
                        llm_model="qwen3.6-35b-a3b-noreason")

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        chunks = [{"id": "test", "object": "chat.completion.chunk", "created": 0,
                   "model": settings.llm_model, "choices": [{"index": 0, "finish_reason": None,
                   "delta": {"content": text}}]} for text in ("L2", " [1]")]
        content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        def factory(**kwargs):
            return ChatOpenAI(**kwargs, http_client=client)

        with patch.object(provider, "settings", settings), patch.object(provider, "ChatOpenAI", side_effect=factory):
            assert "".join(chunk.content for chunk in provider.get_llm().stream("Ridge?")) == "L2 [1]"
