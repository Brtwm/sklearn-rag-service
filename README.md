# scikit-learn docs RAG assistant

Локальный помощник по основам Classic ML. Отвечает на вопросы на русском и английском по документации scikit-learn, показывает источники ответа и выводит текст постепенно в Gradio. Доступны REST API на FastAPI и интерфейс в браузере.

Корпус строится из десяти выбранных страниц документации scikit-learn 1.9 (`linear_model`, `tree`, `model_evaluation`, `ensemble`, `cross_validation`, `preprocessing`, `compose`, `grid_search`, `impute`, `feature_selection`) и локального `data/local/about.md`. `app/scripts/load_corpus.py` сохраняет чанки с URL разделов и стабильными ID в `data/corpus_chunks.jsonl`; `app/scripts/index_corpus.py` загружает их в Qdrant. При запросе сервис находит до четырёх чанков, передаёт их LLM и показывает тот же список источников пользователю.

## Подготовка

Нужны Python 3.11 или новее, Docker с Compose для Qdrant, доступ к интернету для загрузки документации и модели эмбеддингов, а также ключ OpenAI-совместимого LLM-провайдера. По умолчанию настроены Groq, модель `qwen/qwen3.8-27b` и локальный эмбеддер `intfloat/multilingual-e5-small`.

В корне проекта создайте **не коммитируемый** файл `.env`:

```dotenv
LLM_API_KEY=your_provider_key
```

При другом провайдере задайте в нём `LLM_BASE_URL` и `LLM_MODEL`. Не публикуйте действующий ключ.

## Запуск через Anaconda Prompt

Команды выполняются из корня репозитория. Если окружение `sklearn-rag` уже создано, пропустите первую команду.

```bat
conda create -n sklearn-rag python=3.11 -y
conda activate sklearn-rag
python -m pip install -r requirements.txt
docker compose up -d qdrant
python -m app.scripts.load_corpus
python -m app.scripts.index_corpus
python -m uvicorn app.main:app --reload
```

`load_corpus` скачивает десять страниц scikit-learn 1.9, выделяет основные разделы, сохраняет код и создаёт локальный JSONL. `index_corpus` создаёт коллекцию `sklearn_docs_<хеш корпуса>`, проверяет все точки и пробный поиск, затем переключает alias `sklearn_docs`. Повторный запуск с тем же JSONL проверяет уже активную коллекцию без перезаписи точек. Для E5 документы кодируются с `passage: `, запросы — с `query: `; сам текст чанков остаётся без префикса. Если Qdrant ещё запускается, дождитесь его готовности и повторите команду индексации.

## Запуск целиком в Docker

Создайте `.env`, как описано выше, затем из корня репозитория выполните:

```bat
docker compose build app
docker compose up -d qdrant
docker compose run --rm app sh -c "python -m app.scripts.load_corpus && python -m app.scripts.index_corpus"
docker compose up -d app
```

Загрузка корпуса и индексация идут в одном временном контейнере: созданный там JSONL исчезнет после команды, а индекс останется в локальном `qdrant_data_v2/`. Старый `qdrant_data/` сохраняется отдельно. Для повторной индексации повторите команду `docker compose run`: новая коллекция станет активной только после проверки. Оба каталога Qdrant и `.env` не коммитьте.

## Проверка

- Gradio: <http://127.0.0.1:8000/> — ответ поступает по частям, источники отображаются рядом.
- API-документация: <http://127.0.0.1:8000/docs>.
- `GET /health` — процесс работает; не проверяет индекс.
- `GET /ready` — `200`, если активная коллекция доступна и RAG инициализирован; иначе `503`. После появления коллекции сервис может восстановиться без перезапуска.
- `POST /chat` — JSON `{"question": "What is Ridge regression?"}`; ответ содержит `answer` и `sources` (`url`, `snippet`).

Пример запроса из Anaconda Prompt:

```bat
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"What is Ridge regression?\"}"
```

Автоматические проверки без обращения к настоящим Qdrant и LLM:

```bat
python -m pytest -v
```

## Текущие ограничения

- Корпус охватывает десять выбранных тем; поиск пока использует dense-поиск без гибридного поиска и reranker.
- Загрузчик сохраняет основной текст статьи, блоки кода и URL с якорем раздела. После обновления JSONL необходимо повторно запустить `index_corpus`, чтобы новые темы появились в поиске.
- Оценка поиска по разделам будет добавлена следующим шагом; сохранённые результаты в `notebooks/` не являются метриками этой версии корпуса.
- Qdrant server и client закреплены на 1.19.1. Если у вас уже запущен старый Qdrant, `docker compose up -d qdrant` переключит сервис на новый каталог `qdrant_data_v2/`; старый каталог останется на диске.
