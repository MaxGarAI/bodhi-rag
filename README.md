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
├── pinecone_client.py       # Менеджер векторной базы PineconeManager + OpenRouter эмбеддинги + дедупликация
├── bot.py                   # Telegram-бот с поддержкой RAG, сохранением мыслей и защитой от дубликатов
├── chat_buddhism.py         # Интерактивный консольный терминальный чат
├── seed_buddhism_quotes.py  # Скрипт наполнения базы (30 цитат буддизма)
├── test_client.py           # Набор тестов и проверка всех CRUD-операций Pinecone
├── test_deduplication.py    # Тесты косинусного сходства и автоматической фильтрации дубликатов
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
OPENROUTER_API_KEY=sk-or-v1-ваш_ключ_openrouter
PINECONE_KEY=pcsk_ваш_ключ_pinecone
TELEGRAM_KEY=токен_бота_от_BotFather
PINECONE_INDEX=test2
```

> **Важно:** Индекс в Pinecone должен иметь размерность **1536** и метрику **cosine** (соответствует модели `openai/text-embedding-3-small`).

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

После запуска напишите боту в Telegram:
- Задайте вопрос: *«Как научиться отпускать прошлое?»*
- Сохраните новую мысль: *«Запомни: Путь в тысячу ли начинается с одного шага»*
- Проверьте статистику: `/stats`
- Семантический поиск без генерации LLM: `/search осознанное дыхание`

### 3. Запуск консольного диалога (терминал)

```bash
python chat_buddhism.py
```

---

## 🛠 Программный интерфейс (API PineconeManager)

Пример использования [PineconeManager](pinecone_client.py) в собственном коде:

```python
from pinecone_client import PineconeManager, compute_cosine_similarity

# Инициализация (ключи берутся из .env, порог сходства: 0.88)
client = PineconeManager(index_name="test2", duplicate_threshold=0.88)

# 1. Запись текста с автоматической фильтрацией дубликатов по косинусному сходству
res = client.upsert_text(
    id="quote-101",
    text="Спокойствие ума — высшая драгоценность.",
    metadata={"author": "Древняя мудрость", "topic": "Спокойствие"},
    namespace="buddhism",
    filter_duplicates=True,
    duplicate_threshold=0.88,
)

if res["is_duplicate"]:
    print(f"⚠️ Дубликат! Сходство {res['duplicate']['score']:.2%}")
else:
    print("✅ Успешно сохранено!")

# 2. Прямая проверка на наличие дубликата
dup = client.find_duplicate(
    text="Спокойствие ума — это величайшая драгоценность.",
    namespace="buddhism",
    threshold=0.88,
)
if dup:
    print(f"Найден дубликат с ID {dup['id']}, косинусное сходство: {dup['score']:.4f}")

# 3. Семантический поиск по смыслу (с опциональной дедупликацией результатов)
results = client.search_by_text(
    query_text="Что поможет при тревоге?",
    top_k=3,
    namespace="buddhism",
    deduplicate=True,
)

for r in results:
    print(f"[{r['score']:.3f}] {r['metadata']['text']}")

# 4. Чтение вектора по ID
doc = client.fetch_vectors(ids=["quote-101"], namespace="buddhism")

# 5. Удаление
client.delete_vectors(ids=["quote-101"], namespace="buddhism")
```

---

## 📜 Лицензия

MIT License. Свободно для использования, обучения и модификации.
