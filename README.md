# 🌸 Bodhi RAG (Лотос Мудрости)

Интеллектуальная система семантического поиска и RAG-ассистент по философии буддизма, осознанности и медитации на базе векторной базы данных **Pinecone**, моделей **OpenRouter** и интерфейса **Telegram-бота**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-green.svg)](https://www.pinecone.io/)
[![OpenRouter](https://img.shields.io/badge/Embeddings%20%26%20LLM-OpenRouter-purple.svg)](https://openrouter.ai/)
[![Telegram](https://img.shields.io/badge/Bot-python--telegram--bot-2CA5E0.svg)](https://core.telegram.org/bots)

---

## 🌟 Ключевые возможности

- 🧠 **Семантический поиск (Pinecone)**: находит учения и цитаты по глубинному смыслу вопроса, даже если в вопросе нет совпадающих ключевых слов.
- 💬 **RAG-генерация ответов (OpenRouter LLM)**: языковая модель формулирует мудрые, практичные и сострадательные ответы, опираясь на найденные первоисточники из базы знаний.
- ✍️ **Запоминание новых фраз на лету**: бот распознает естественные речевые триггеры (`Запомни: ...`, `Запиши: ...`, `Добавь: ...`, `Сохрани: ...`) или команду `/add`, векторизует мысль и мгновенно сохраняет в Pinecone.
- 🛡️ **Автоматическая фильтрация дубликатов (Cosine Similarity)**: встроенная в `PineconeManager` проверка косинусного сходства векторов с настраиваемым порогом (по умолчанию `0.88`). Предотвращает засорение базы знаний повторными или близкими по смыслу мыслями как при добавлении в боте, так и при пакетной загрузке.
- 🤖 **Telegram-бот**: удобный асинхронный интерфейс с командами `/search`, `/stats`, `/add`, `/help`.
- 💻 **Консольный режим**: автономный скрипт интерактивного диалога в терминале.
- 📦 **Модульный менеджер `PineconeManager` (`PineconeVectorClient`)**:
  - Полная типизация и docstring-документация.
  - Дедупликация на основе математического расчета косинусного сходства (`compute_cosine_similarity`, `find_duplicate`).
  - Поддержка операций: подключение, `upsert` (одиночный и пакетный с дедупликацией), `fetch`, `query/search`, `delete`.
  - Встроенная интеграция с эмбеддингами OpenRouter (`text-embedding-3-small`, 1536 dim).

---

## 📁 Структура проекта

```text
bodhi-rag/
├── pinecone_client.py       # Менеджер векторной базы PineconeManager + OpenRouter/OpenAI эмбеддинги + дедупликация
├── bot.py                   # Telegram-бот с авто-сохранением мыслей, дедупликацией и RAG
├── chat_buddhism.py         # Интерактивный консольный терминальный чат
├── seed_buddhism_quotes.py  # Скрипт наполнения базы (30 цитат буддизма)
├── test_client.py           # Набор тестов и проверка всех CRUD-операций Pinecone
├── test_deduplication.py    # Тесты косинусного сходства, дедупликации и логирования actions
├── deduplication.log        # Зафиксированный лог дедупликации (action: created / skipped / updated)
├── requirements.txt         # Список Python-зависимостей
├── .env.example             # Шаблон переменных окружения
├── .gitignore               # Исключение секретов (.env) и виртуального окружения
└── README.md                # Документация проекта
```

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/MaxGarAI/bodhi-rag.git
cd bodhi-rag
```

### 2. Создание и активация виртуального окружения

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения

Скопируйте файл `.env.example` в `.env` и укажите ваши ключи:

```bash
cp .env.example .env
```

Заполните `.env`:
```env
# Базовый URL для OpenAI-совместимого API (OpenRouter, OpenAI или локальный прокси)
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# API-ключ для эмбеддингов и LLM (поддерживаются OPENAI_API_KEY и OPENROUTER_API_KEY)
OPENROUTER_API_KEY=sk-or-v1-ваш_ключ_openrouter
OPENAI_API_KEY=sk-or-v1-ваш_ключ_openrouter

# Pinecone API Key и индекс
PINECONE_KEY=pcsk_ваш_ключ_pinecone
PINECONE_INDEX=test2

# Telegram Bot Token (для запуска бота)
TELEGRAM_KEY=токен_бота_от_BotFather

# Порог косинусного сходства (Cosine Similarity) для дедупликации (0.0 - 1.0)
DUPLICATE_THRESHOLD=0.88

# Файл логирования результатов дедупликации
LOG_FILE=deduplication.log
```

> **Важно:** 
> - Поддерживается настройка `OPENAI_BASE_URL` через конструктор `PineconeManager(openai_base_url=...)` и переменную окружения `OPENAI_BASE_URL` (по умолчанию `https://openrouter.ai/api/v1`).
> - Индекс в Pinecone должен иметь размерность **1536** и метрику **cosine** (соответствует модели `openai/text-embedding-3-small`).

---

## 📖 Использование

### 1. Первичное наполнение базы знаний (30 цитат)

Запустите скрипт для пакетной векторизации и загрузки базовых текстов в раздел `buddhism`:

```bash
python seed_buddhism_quotes.py
```

### 2. Запуск Telegram-бота

```bash
python bot.py
```

В боте реализована интеллектуальная маршрутизация сообщений:
- **Обычные утверждения и мысли пользователя** автоматически направляются на **сохранение с проверкой дедупликации** (а также через триггеры `Запомни: ...`, `Запиши: ...` или команду `/add`). Если фраза уже есть или очень похожа по смыслу на существующую, бот сообщает об обнаружении дубликата и предотвращает засорение базы знаний.
- **Вопросы** (содержащие знак `?`, вопросительные слова или команду `/ask`) направляются в RAG-пайплайн для семантического поиска и генерации ответа.
- Проверка статистики: `/stats`
- Семантический поиск цитат без генерации LLM: `/search осознанное дыхание`

### 3. Запуск консольного диалога (терминал)

```bash
python chat_buddhism.py
```

### 4. Запуск тестов дедупликации

```bash
python test_deduplication.py
```

---

## 📋 Результаты логирования дедупликации (`action: updated` / `action: skipped`)

Все операции сохранения текста с проверкой дедупликации логируются с явным указанием действия (`action: created`, `action: skipped`, `action: updated`) как в стандартный поток вывода, так и в файл [deduplication.log](deduplication.log).

### Реальный лог работы дедупликации:

```log
2026-10-09 11:18:41,093 - INFO - Клиент эмбеддингов инициализирован (base_url: 'https://openrouter.ai/api/v1', модель: 'openai/text-embedding-3-small')
2026-10-09 11:18:45,530 - INFO - Успешно подключено к индексу Pinecone: 'test2'
2026-10-09 11:18:49,254 - INFO - Успешно записано 1 векторов в индекс 'test2' (пакетов: 1, namespace: 'test_dedup_1791533921').
2026-10-09 11:18:49,254 - INFO - action: created | id: 'test-orig-1791533925' | text: 'Истинный покой обретается в осознанности и тишине сердца 1791533925'
2026-10-09 11:18:51,690 - INFO - action: skipped | duplicate of id: 'test-orig-1791533925' | similarity: 0.9715 >= threshold: 0.88 | text: 'Истинный покой обретается в осознанности и тишине сердца! 1791533925'
2026-10-09 11:18:53,229 - INFO - Успешно записано 1 векторов в индекс 'test2' (пакетов: 1, namespace: 'test_dedup_1791533921').
2026-10-09 11:18:53,230 - INFO - action: updated | id: 'test-orig-1791533925' | similarity: 0.9795 >= threshold: 0.88 | text: 'Истинный покой обретается в осознанности и глубокой тишине сердца 1791533925'
2026-10-09 11:18:54,347 - INFO - action: created | id: 'b1-1791533925' | text: 'Ум — это всё. Чем ты думаешь, тем ты и становишься 1791533925.'
2026-10-09 11:18:54,348 - INFO - action: skipped | in_batch_duplicate of 'b1-1791533925' | similarity: 0.9575 >= 0.88 | text: 'Ум — это абсолютно всё. Чем ты думаешь, тем ты и становишься 1791533925.'
```

### Формат записей действий:
- **`action: created`**: Текст уникален, вектор успешно добавлен в Pinecone.
- **`action: skipped`**: Найдена семантически похожая фраза с косинусным сходством $\ge$ порога (`duplicate_threshold`). При стратегии `on_duplicate="skip"` добавление дубликата отклонено.
- **`action: updated`**: Найдена семантически похожая фраза. При стратегии `on_duplicate="update"` существующая запись обновлена новым текстом и вектором.

---

## 🛠 Программный интерфейс (API PineconeManager)

Пример использования [PineconeManager](pinecone_client.py) в собственном коде:

```python
from pinecone_client import PineconeManager, compute_cosine_similarity

# Инициализация с поддержкой OPENAI_BASE_URL (по умолчанию OpenRouter или OpenAI)
client = PineconeManager(
    index_name="test2",
    openai_base_url="https://openrouter.ai/api/v1",
    duplicate_threshold=0.88,
)

# 1. Запись с пропуском дубликата (action: skipped)
res_skip = client.upsert_text(
    id="quote-101",
    text="Спокойствие ума — высшая драгоценность.",
    namespace="buddhism",
    filter_duplicates=True,
    on_duplicate="skip",
)
print(f"Action: {res_skip['action']}")  # 'created' или 'skipped'

# 2. Запись с обновлением существующей похожей записи (action: updated)
res_update = client.upsert_text(
    id="quote-102",
    text="Спокойствие ума — это величайшая драгоценность.",
    namespace="buddhism",
    filter_duplicates=True,
    on_duplicate="update",
)
print(f"Action: {res_update['action']}")  # 'created' или 'updated'

# 3. Прямой поиск дубликата
dup = client.find_duplicate(
    text="Спокойствие ума — это величайшая драгоценность.",
    namespace="buddhism",
    threshold=0.88,
)
if dup:
    print(f"Найден дубликат с ID {dup['id']}, косинусное сходство: {dup['score']:.4f}")

# 4. Семантический поиск по смыслу
results = client.search_by_text(
    query_text="Что поможет при тревоге?",
    top_k=3,
    namespace="buddhism",
    deduplicate=True,
)
for r in results:
    print(f"[{r['score']:.3f}] {r['metadata']['text']}")

# 5. Удаление
client.delete_vectors(ids=["quote-101"], namespace="buddhism")
```

---

## 📜 Лицензия

MIT License. Свободно для использования, обучения и модификации.

