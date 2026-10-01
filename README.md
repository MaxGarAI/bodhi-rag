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
- 🤖 **Telegram-бот**: удобный асинхронный интерфейс с командами `/search`, `/stats`, `/add`, `/help`.
- 💻 **Консольный режим**: автономный скрипт интерактивного диалога в терминале.
- 📦 **Модульный клиент `PineconeVectorClient`**:
  - Полная типизация и docstring-документация.
  - Поддержка операций: подключение, `upsert` (одиночный и пакетный), `fetch`, `query/search`, `delete`.
  - Встроенная интеграция с эмбеддингами OpenRouter (`text-embedding-3-small`, 1536 dim).

---

## 📁 Структура проекта

```text
bodhi-rag/
├── pinecone_client.py       # Клиент векторной базы Pinecone + OpenRouter эмбеддинги
├── bot.py                   # Telegram-бот с поддержкой RAG и сохранением мыслей
├── chat_buddhism.py         # Интерактивный консольный терминальный чат
├── seed_buddhism_quotes.py  # Скрипт наполнения базы (30 цитат буддизма)
├── test_client.py           # Набор тестов и проверка всех CRUD-операций Pinecone
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

## 🛠 Программный интерфейс (API клиента)

Пример использования [PineconeVectorClient](pinecone_client.py) в собственном коде:

```python
from pinecone_client import PineconeVectorClient

# Инициализация (ключи берутся из .env)
client = PineconeVectorClient(index_name="test2")

# 1. Запись текста с автоматической генерацией эмбеддинга
client.upsert_text(
    id="quote-101",
    text="Спокойствие ума — высшая драгоценность.",
    metadata={"author": "Древняя мудрость", "topic": "Спокойствие"},
    namespace="buddhism"
)

# 2. Семантический поиск по смыслу
results = client.search_by_text(
    query_text="Что поможет при тревоге?",
    top_k=3,
    namespace="buddhism"
)

for r in results:
    print(f"[{r['score']:.3f}] {r['metadata']['text']}")

# 3. Чтение вектора по ID
doc = client.fetch_vectors(ids=["quote-101"], namespace="buddhism")

# 4. Удаление
client.delete_vectors(ids=["quote-101"], namespace="buddhism")
```

---

## 📜 Лицензия

MIT License. Свободно для использования, обучения и модификации.
