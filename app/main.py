from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from fastapi import FastAPI, Request

from app.schemas.predict import PredictRequest, PredictResponse

MODEL_PATH = Path("models/wine_model.pkl")
CLASS_NAMES = ["class_0", "class_1", "class_2"]
FEATURE_ORDER = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", 
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", 
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
]

def _build_gradio_demo(model) -> gr.Blocks:
    default_values = [13.5, 1.8, 2.4, 18.0, 105.0, 2.7, 2.9, 0.28, 1.85, 5.5, 1.05, 3.2, 1180.0]

    # Три предзаполненных примера — по одному для каждого класса.
    # Числа взяты из реальных строк датасета load_wine.
    examples = [
        [13.50, 1.81, 2.61, 20.0, 96.0, 2.53, 2.61, 0.28, 1.66, 3.52, 1.12, 3.82, 845],   # class_0 typical
        [12.37, 0.94, 1.36, 10.6, 88.0, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520],   # class_1 typical
        [13.71, 5.65, 2.45, 20.5, 95.0, 1.68, 0.61, 0.52, 1.06, 7.70, 0.64, 1.74, 740],   # class_2 typical
    ]

    def predict_wine(*features):
        row = pd.DataFrame([dict(zip(FEATURE_ORDER, features))])[FEATURE_ORDER]
        proba = model.predict_proba(row)[0]
        return {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}

    with gr.Blocks(title="Wine classifier") as demo:
        gr.Markdown(
            "# 🍷 Wine classifier\n"
            "ML-эндпоинт: по 13 химическим показателям партии вина определяет сорт винограда. "
            "Можно ввести значения вручную или нажать на один из примеров под формой."
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Вход — химический анализ партии")
                inputs = [
                    gr.Number(label=name, value=v)
                    for name, v in zip(FEATURE_ORDER, default_values)
                ]
                submit = gr.Button("Предсказать сорт", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Выход — предсказание модели")
                output = gr.Label(label="Распределение вероятностей по классам")

        gr.Examples(
            examples=examples,
            inputs=inputs,
            label="Примеры партий (клик заполняет поля)",
        )

        submit.click(fn=predict_wine, inputs=inputs, outputs=output)

    return demo

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Грузим модель один раз при старте
    app.state.model = joblib.load(MODEL_PATH)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    model = request.app.state.model

    # Превращаем Pydantic-объект в DataFrame с теми же именами колонок,
    # которые модель видела на обучении. by_alias=True вернёт
    # оригинальное имя od280/od315_of_diluted_wines со слэшем.
    row = pd.DataFrame([req.model_dump(by_alias=True)])
    row = row[FEATURE_ORDER]

    proba = model.predict_proba(row)[0]
    pred_idx = int(proba.argmax())

    return PredictResponse(
        predicted_class=CLASS_NAMES[pred_idx],
        probabilities={CLASS_NAMES[i]: float(round(p, 4)) for i, p in enumerate(proba)}
    )

_model = joblib.load(MODEL_PATH)
demo = _build_gradio_demo(_model)
app = gr.mount_gradio_app(app, demo, path="/")