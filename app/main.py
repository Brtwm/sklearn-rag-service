import time
from contextlib import asynccontextmanager
from threading import Lock

import gradio as gr
from fastapi import FastAPI, HTTPException

from app.rag.chain import build_rag_chain, format_docs_with_sources, index_available
from app.schemas.chat import ChatRequest, ChatResponse, Source

_chain = None
_retriever = None
_rag_lock = Lock()

# LaTeX delimiters для Gradio Chatbot. LLM-ответы про Ridge, Lasso, метрики
# содержат формулы $$..$$ / \[..\] / $..$ — без этого блока они отрисуются
# как сырые строки `$\ell_1$`.
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever
    _ensure_ready()
    yield
    _chain = None
    _retriever = None


def _ensure_ready() -> bool:
    global _chain, _retriever
    with _rag_lock:
        try:
            if not index_available():
                _chain = None
                _retriever = None
                return False
            if _chain is None or _retriever is None:
                _chain, _retriever = build_rag_chain()
            return True
        except Exception:
            _chain = None
            _retriever = None
            return False


app = FastAPI(title="RAG service", lifespan=lifespan)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/ready")
def ready() -> dict[str, str]:
    if not _ensure_ready():
        raise HTTPException(status_code=503, detail="Active index unavailable")
    return {"status": "ready"}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not _ensure_ready():
        raise HTTPException(status_code=503, detail="Active index unavailable")
    retriever, chain = _retriever, _chain
    try:
        docs = retriever.invoke(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval temporarily unavailable. Raw: {type(exc).__name__}",
        ) from exc
    try:
        answer = chain.invoke({
            "question": payload.question,
            "context": format_docs_with_sources(docs),
        })
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM provider temporarily unavailable. "
                f"Try again in 30-60 seconds. Raw: {type(exc).__name__}"
            ),
        ) from exc
    sources = [
        Source(
            url=doc.metadata.get("source", "unknown"),
            snippet=doc.page_content[:200].strip(),
        )
        for doc in docs
    ]
    return ChatResponse(answer=answer, sources=sources)


def _format_timings(retrieval_ms: float, llm_ms: float | None, llm_error: str | None) -> str:
    lines = [
        "### ⏱ Тайминги последнего запроса",
        "",
        f"- 🔍 **Retrieval (embed + Qdrant):** {retrieval_ms:.0f} ms",
    ]
    if llm_ms is not None:
        lines.append(f"- 🤖 **LLM call:** {llm_ms:.0f} ms")
        lines.append(f"- 📊 **Total:** {retrieval_ms + llm_ms:.0f} ms")
    else:
        lines.append(f"- 🤖 **LLM call:** ❌ {llm_error}")
    return "\n".join(lines)

def _format_sources(docs: list) -> str:
    if not docs:
        return "### 📚 Источники\n\n_Ничего не найдено_"
    lines = ["### 📚 Источники", ""]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:140].strip().replace("\n", " ")
        lines.append(f"**[{i}]** `{source}`")
        lines.append(f"> {snippet}…")
        lines.append("")
    return "\n".join(lines)


def respond(message: str, history: list):
    """Streaming Gradio handler — generator that yields on every chunk."""
    if not message or not message.strip():
        yield history, "", "### ⏱ Тайминги\n\n_Пустой запрос_", "### 📚 Источники\n\n_—_"
        return

    history = history + [{"role": "user", "content": message}]

    if not _ensure_ready():
        history.append({"role": "assistant", "content": "⚠️ Индекс пока недоступен. Попробуй позже."})
        yield history, "", "### ⏱ Тайминги\n\n_Индекс недоступен_", "### 📚 Источники\n\n_—_"
        return

    retriever, chain = _retriever, _chain

    t0 = time.perf_counter()
    try:
        docs = retriever.invoke(message)
    except Exception as exc:
        history.append({"role": "assistant", "content": f"⚠️ Поиск сейчас недоступен ({type(exc).__name__})."})
        yield history, "", "### ⏱ Тайминги\n\n_Ошибка поиска_", "### 📚 Источники\n\n_—_"
        return
    retrieval_ms = (time.perf_counter() - t0) * 1000
    sources_panel = _format_sources(docs)

    # Yield #1: sources уже на экране, LLM ещё не начал писать.
    history.append({"role": "assistant", "content": ""})
    yield (
        history, "",
        "### ⏱ Тайминги\n\n"
        f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
        "- 🤖 **LLM:** _streaming…_",
        sources_panel,
    )

    t1 = time.perf_counter()
    ttft_ms: float | None = None
    accumulated = ""
    try:
        for chunk in chain.stream({
            "question": message,
            "context": format_docs_with_sources(docs),
        }):
            if not chunk:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t1) * 1000
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield (
                history, "",
                f"### ⏱ Тайминги\n\n"
                f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
                f"- ⚡ **TTFT (1st token):** {ttft_ms:.0f} ms\n"
                f"- 🤖 **LLM:** _streaming… {len(accumulated)} chars_",
                sources_panel,
            )
        
        llm_total_ms = (time.perf_counter() - t1) * 1000
        yield (
            history, "",
            "### ⏱ Тайминги последнего запроса\n\n"
            f"- 🔍 **Retrieval:** {retrieval_ms:.0f} ms\n"
            f"- ⚡ **TTFT:** {ttft_ms:.0f} ms\n"
            f"- 🤖 **LLM stream (full):** {llm_total_ms:.0f} ms\n"
            f"- 📊 **Total:** {retrieval_ms + llm_total_ms:.0f} ms",
            sources_panel,
        )

    except Exception as exc:
        history[-1]["content"] = (
            f"⚠️ LLM-провайдер сейчас недоступен ({type(exc).__name__}). "
            f"Попробуй через 30-60 секунд."
        )
        yield (
            history, "",
            _format_timings(retrieval_ms, None, type(exc).__name__),
            sources_panel,
        )


CSS = """
.gradio-container { max-width: 100% !important; padding: 1rem !important; }
#chatbot { height: calc(100vh - 220px) !important; min-height: 500px !important; }
#side-panel { height: calc(100vh - 220px) !important; overflow-y: auto !important;
              padding: 1rem !important; border-left: 1px solid #ddd !important; }
"""

with gr.Blocks(
    title="scikit-learn docs RAG",
    fill_height=True,
) as demo:
    gr.Markdown(
        "# 📖 scikit-learn docs RAG assistant\n"
        "_Спрашивай про Linear models, Decision trees, Metrics — на русском или английском._"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                latex_delimiters=LATEX_DELIMITERS,
                buttons=["copy"],
                avatar_images=(None, None),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Например: «Покажи формулу Ridge» или «Чем precision отличается от recall»",
                    scale=8,
                    container=False,
                    autofocus=True,
                )
                send = gr.Button("Отправить", scale=1, variant="primary")
            gr.Examples(
                examples=[
                    "How does Ridge regression work?",
                    "Что ты умеешь?",
                    "Объясни разницу между precision и recall с формулами",
                    "When does a decision tree overfit?",
                ],
                inputs=msg,
            )
            with gr.Column(scale=1, elem_id="side-panel"):
                timings_md = gr.Markdown(
                    "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
                )
                sources_md = gr.Markdown("### 📚 Источники\n\n_—_")

    msg.submit(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])
    send.click(respond, [msg, chatbot], [chatbot, msg, timings_md, sources_md])


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    theme=gr.themes.Soft(),
    css=CSS,
)
