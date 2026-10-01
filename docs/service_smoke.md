# Smoke-test сервиса без LLM-квоты

Команды выполняются из корня проекта. Для рабочего индекса в Anaconda Prompt:

```bat
conda activate sklearn-rag
python -m app.scripts.smoke_service
```

Нужны работающий Qdrant, активный hybrid-индекс, локальный
`data/corpus_chunks.jsonl` и веса E5-small/MiniLM. Режим — `hybrid_rerank`,
`TOP_K=4`. Ключ провайдера не нужен: сценарий устанавливает тестовый ключ
только в своём процессе, заменяет генерацию локальной функцией и направляет
настройку LLM на недоступный loopback-порт. Настройки `.env` не изменяются.

Для каждого уровня параллельности (1 и 4) запускается отдельный процесс
настоящего FastAPI/Gradio-приложения с CPU-моделями. После двух прогревочных
HTTP-запросов выполняются три прохода по восьми вопросам (4 EN, 4 RU).
Заглушка возвращает исходный вопрос и точный контекст генерации: проверяются
принадлежность ответа запросу, текст чанков, порядок источников и отсутствие
дублирующихся чанков. Две независимые сессии официального клиента Gradio
отправляются одновременно; проверяются очередь, источники, состояние завершения
и восстановление элементов ввода. Это проверка протокола Gradio, без автоматизации
браузерного отображения.

Упорядоченные ID чанков из первого последовательного прохода служат
эталоном для каждого вопроса. Остальные проходы и сессии Gradio должны
вернуть те же ID; так обнаруживается подмена контекста контекстом другого
запроса, даже если все его чанки действительно существуют в корпусе.

Отчёт: `notebooks/service_smoke_v1.json`. Успешный прогон завершается с кодом 0;
ошибки запросов и проверки очереди сохраняются в отчёте и дают код 1. Ошибка
подготовки/старта не заменяет предыдущий отчёт. Для отдельного повтора:

```bat
python -m app.scripts.smoke_service --output notebooks/service_smoke_repeat.json
```

Задержка — полное время HTTP `/chat`, включая проверку готовности, настоящий
поиск и заглушку. p50/p95 рассчитаны линейной интерполяцией по 24 измеренным
запросам для каждого уровня, включая ошибочные запросы, если они были.
Загрузка моделей, прогрев и Gradio-запросы не входят в эти задержки.
Память — пик всего процесса сервера от импорта/загрузки до завершения REST
и Gradio-проверок; клиентский процесс и Qdrant в неё не входят.
Для Qdrant отдельно сохраняется снимок `docker stats`, **не пик**.
При другом имени контейнера задайте `--qdrant-container NAME`; если Docker
недоступен, поле памяти Qdrant остаётся `null`.

Сценарий не индексирует данные и не переключает alias. Рабочий alias проверяется
до и после прогона. Генерация заглушкой не имитирует задержку или качество
настоящего LLM, а тест не устанавливает SLA для приложения.

## Проверка Docker-сборки и пустого индекса

Используются отдельные имена контейнеров и сети. Если эти имена уже заняты,
выберите новые; не удаляйте существующие контейнеры для повторного запуска.
Команды ниже предназначены для Anaconda Prompt, не PowerShell.

```bat
docker build --no-cache -t sklearn-rag-step5-check .
docker network create sklearn-rag-step5-check
docker run -d --rm --name sklearn-rag-step5-qdrant --network sklearn-rag-step5-check qdrant/qdrant:v1.19.1
```

Временный Qdrant использует своё пустое хранилище внутри контейнера;
`qdrant_data/`, `qdrant_data_v2/` и рабочий alias не затрагиваются.
После готовности Qdrant проверить поведение без индекса:

```bat
docker run --rm --network sklearn-rag-step5-check -e LLM_API_KEY=local-smoke-only -e QDRANT_URL=http://sklearn-rag-step5-qdrant:6333 sklearn-rag-step5-check python -c "from fastapi.testclient import TestClient; from app.main import app; c=TestClient(app); print({p:c.get(p).status_code for p in ['/health','/ready','/']}); assert c.get('/health').status_code==200; assert c.get('/ready').status_code==503; assert c.post('/chat',json={'question':'Ridge?'}).status_code==503"
```

Создать новую папку результатов (не использовать папку предыдущего прогона),
скопировать туда версионированный набор вопросов и выполнить загрузку,
индексацию и полный smoke-test в новом контейнере:

```bat
set CHECK_DIR=%TEMP%\sklearn-rag-step5-check-%RANDOM%
mkdir "%CHECK_DIR%"
copy data\eval\retrieval_questions_v1.json "%CHECK_DIR%\retrieval_questions_v1.json"
docker run --rm --name sklearn-rag-step5-app --network sklearn-rag-step5-check -e LLM_API_KEY=local-smoke-only -e QDRANT_URL=http://sklearn-rag-step5-qdrant:6333 -v "%CHECK_DIR%:/check" sklearn-rag-step5-check sh -c "python -m app.scripts.load_corpus && python -m app.scripts.index_corpus --device cpu && python -m app.scripts.smoke_service --questions /check/retrieval_questions_v1.json --output /check/fresh_setup_smoke_v1.json --qdrant-container sklearn-rag-step5-qdrant"
```

Результат находится в `%CHECK_DIR%\fresh_setup_smoke_v1.json`.
В контейнере приложения нет Docker CLI, поэтому снимок памяти Qdrant
следует получить отдельно с хоста:

```bat
docker stats --no-stream --format "{{.MemUsage}}" sklearn-rag-step5-qdrant
docker stop sklearn-rag-step5-qdrant
docker network rm sklearn-rag-step5-check
```

Остановка удаляет только созданный временный Qdrant (`--rm`). Образ и папка
результатов остаются для просмотра. Сохранённый проверенный Docker-прогон
этого этапа: `notebooks/fresh_setup_smoke_v1.json`.

## Сохранённые результаты — 30 сентября 2026

Машина: Intel Core i7-12700H, 14 ядер / 20 логических процессоров,
16,87 ГБ установленной RAM; Docker ограничен 7,61 ГиБ. Во время замеров
работали другие службы разработки; это локальный smoke-test без выделенной машины.
В каждом окружении завершены все 48 измеренных запросов, четыре прогревочных
и четыре запроса Gradio. Ошибок нет, максимум одновременно выполняемых
Gradio-запросов — один, alias сохранён.

| Окружение | Параллельность HTTP | p50, мс | p95, мс | Пик процесса сервиса, МиБ |
|---|---:|---:|---:|---:|
| Anaconda, Python 3.12.14 | 1 | 1956,9 | 2382,8 | 2327,9 |
| Anaconda, Python 3.12.14 | 4 | 6895,4 | 7648,1 | 2460,3 |
| Свежий Docker, Python 3.11.16 | 1 | 2055,2 | 2582,6 | 2038,7 |
| Свежий Docker, Python 3.11.16 | 4 | 8973,1 | 9454,0 | 2048,9 |

Отчёты: [Anaconda](../notebooks/service_smoke_v1.json) и
[Docker с индексацией с нуля](../notebooks/fresh_setup_smoke_v1.json).
Версии библиотек, ревизии моделей, хеши входов, отдельные задержки
и снимки памяти Qdrant записаны в отчётах. Результаты окружений не объединены.
При четырёх запросах CPU-поиск увеличивает задержку; порог SLA не установлен.
Насыщение всех восьми мест очереди не измерялось: её конфигурация проверяется
обычным тестом, а последовательное выполнение — двумя реальными сессиями.

При проверке Docker зависимости установлены командой `docker build --no-cache`.
После исправлений сценария обновлён только слой исходников с сохранением этой
свежей установки. Корпус скачан и 704 чанка проиндексированы заново;
модели загружены без подключённого кеша хоста. Пустой индекс вернул
`/health=200`, `/ready=503`, `/chat=503`; после индексации проверки прошли.
Временный Qdrant и сеть остановлены и удалены после проверки.

Проверка `python -m pytest -q`: 139 тестов прошли в Anaconda и 139 —
в свежем Docker-образе. Обычные тесты самого сценария используют заглушки;
они не запускают Docker, не загружают модели и не обращаются к провайдеру.
