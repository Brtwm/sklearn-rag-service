from fastapi import FastAPI

app = FastAPI(title="RAG service")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}