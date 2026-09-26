# scikit-learn docs RAG assistant

Локальный помощник по основам Classic ML. Отвечает на вопросы на русском и английском по документации scikit-learn, показывает источники ответа и выводит текст постепенно в Gradio. Доступны REST API на FastAPI и интерфейс в браузере.

Сейчас корпус строится из трёх страниц документации (`linear_model`, `tree`, `model_evaluation`) и локального `data/local/about.md`. `app/scripts/load_corpus.py` сохраняет чанки в `data/corpus_chunks.jsonl`; `app/scripts/index_corpus.py` загружает их в Qdrant. При запросе сервис находит до четырёх чанков, передаёт их LLM и показывает тот же список источников пользователю.

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

`load_corpus` скачивает страницы scikit-learn и создаёт локальный JSONL. `index_corpus` **удаляет и пересоздаёт** коллекцию `sklearn_docs`, затем загружает чанки и выполняет пробные поисковые запросы. При первом запуске также скачивается модель эмбеддингов. Если Qdrant ещё запускается, дождитесь его готовности и повторите команду индексации.

## Запуск целиком в Docker

Создайте `.env`, как описано выше, затем из корня репозитория выполните:

```bat
docker compose build app
docker compose up -d qdrant
docker compose run --rm app sh -c "python -m app.scripts.load_corpus && python -m app.scripts.index_corpus"
docker compose up -d app
```

Загрузка корпуса и индексация идут в одном временном контейнере: созданный там JSONL исчезнет после команды, а индекс останется в локальном `qdrant_data/`. Для повторной индексации остановите приложение и повторите команду `docker compose run`; она пересоздаст активную коллекцию. Каталог `qdrant_data/` и `.env` не коммитьте.

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

- Корпус охватывает только три темы; поиск пока использует dense-поиск без гибридного поиска и reranker.
- HTML-загрузка может захватывать элементы навигации и разрывать код; ссылки ведут на страницу документации, а не на конкретный раздел.
- Для E5 пока не применяются префиксы `query:` и `passage:`. Качество поиска и оценка будут доработаны отдельно; сохранённые результаты в `notebooks/` не являются метриками этой версии корпуса.
- `docker-compose.yml` использует Qdrant 1.12.0, а незакреплённый `qdrant-client` может установиться более новой версии и вывести предупреждение о совместимости. Согласование версий запланировано вместе с обновлением индекса.
