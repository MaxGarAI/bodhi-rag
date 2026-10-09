"""
Telegram-бот «Лотос Мудрости» на базе Pinecone и OpenRouter.

Возможности:
1. Ответы на жизненные и философские вопросы с помощью RAG (Pinecone + OpenRouter LLM).
2. Запоминание новых мыслей/цитат естественным языком:
   - «Запомни: ...»
   - «Запиши: ...»
   - «Добавь: ...»
   - «Сохрани: ...»
   или командой /add <текст>.
3. Команды: /start, /help, /stats, /search <запрос>.
"""

import os
import time
import logging
import asyncio
from dotenv import load_dotenv

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from pinecone_client import PineconeManager, PineconeVectorClient

# Загружаем переменные из .env
load_dotenv()

# Настройка логирования: консоль + deduplication.log
LOG_FILE = os.getenv("LOG_FILE", "deduplication.log")
logger = logging.getLogger("bodhi_rag")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _file_h = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _file_h.setFormatter(_formatter)
    logger.addHandler(_file_h)
    _stream_h = logging.StreamHandler()
    _stream_h.setFormatter(_formatter)
    logger.addHandler(_stream_h)

# Инициализация переменных окружения
TELEGRAM_KEY = os.getenv("TELEGRAM_KEY") or os.getenv("TELEGRAM_BOT_TOKEN")
INDEX_NAME = os.getenv("PINECONE_INDEX", "test2")
NAMESPACE = "buddhism"

if not TELEGRAM_KEY:
    raise ValueError("TELEGRAM_KEY не найден в файле .env!")

# Инициализация менеджера Pinecone
pinecone_client = PineconeManager(index_name=INDEX_NAME)

# Триггеры для сохранения новых мыслей
SAVE_TRIGGERS = (
    "запомни",
    "запиши",
    "добавь",
    "сохрани",
    "зафиксируй",
    "remember",
    "save",
)


def extract_save_text(message_text: str) -> str:
    """
    Извлекает текст для сохранения, если сообщение начинается с одного из триггеров.
    """
    lower = message_text.lower()
    for trigger in SAVE_TRIGGERS:
        if lower.startswith(trigger):
            remaining = message_text[len(trigger) :].strip()
            # Убираем знаки препинания в начале фразы (: , -)
            if remaining and remaining[0] in [":", "-", ",", "—"]:
                remaining = remaining[1:].strip()
            return remaining
    return ""


def is_question_message(text: str) -> bool:
    """
    Определяет, является ли сообщение вопросом для RAG-ответа,
    или утверждением/мыслью для сохранения в Pinecone с проверкой дедупликации.
    """
    stripped = text.strip()
    if stripped.endswith("?"):
        return True

    lower = stripped.lower()
    question_starters = (
        "как ", "что ", "почему ", "зачем ", "где ", "когда ", "кто ", "куда ",
        "откуда ", "какой ", "какая ", "какие ", "какое ", "сколько ",
        "подскажи", "расскажи", "объясни", "посоветуй", "помоги", "в чем ",
        "правда ли", "можно ли", "стоит ли", "бывает ли"
    )
    return any(lower.startswith(q) for q in question_starters)


# ------------------------------------------------------------------------------
# Обработчики команд
# ------------------------------------------------------------------------------


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /start — приветствие и краткое руководство.
    """
    first_name = update.effective_user.first_name if update.effective_user else "друг"
    welcome_text = (
        f"🙏 <b>Приветствую, {first_name}!</b>\n\n"
        "Я — бот-наставник по <b>буддийской философии и осознанности</b>.\n"
        "Моя база знаний сохранена в векторной базе <b>Pinecone</b> и готова помочь вам найти внутренний покой.\n\n"
        "<b>Что я умею:</b>\n"
        "1. <b>Сохранять новые мысли и цитаты:</b>\n"
        "   • Просто отправьте фразу в чат (например: <code>Спокойствие рождается изнутри</code>) "
        "или напишите <code>Запомни: ...</code> / команду <code>/add ...</code>.\n"
        "   • Включена <b>автоматическая дедупликация</b> (Cosine Similarity): дубликаты отсекаются с логированием <code>action: skipped</code> или <code>action: updated</code>!\n\n"
        "2. <b>Отвечать на вопросы (RAG):</b>\n"
        "   • Задайте вопрос со знаком вопроса <b>?</b> (например: <i>«Как справиться с тревогой?»</i>) "
        "или используйте команду <code>/ask &lt;вопрос&gt;</code>.\n\n"
        "<b>Команды:</b>\n"
        "• /ask &lt;вопрос&gt; — задать вопрос мудрецу (RAG-генерация)\n"
        "• /add &lt;текст&gt; — сохранить цитату в базу\n"
        "• /search &lt;запрос&gt; — чистый векторный поиск похожих цитат\n"
        "• /stats — количество векторов и статус базы знаний\n"
        "• /help — подробная справка"
    )
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.HTML)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /help.
    """
    help_text = (
        "📖 <b>Как пользоваться ботом:</b>\n\n"
        "• <b>Добавить мудрость:</b> Просто отправьте мне цитату или мысль (или используйте <code>/add ...</code> / <i>«Запомни: ...»</i>). "
        "Бот проверит её на дубликаты в Pinecone: уникальные сохраняются (<code>action: created</code>), повторы отсекаются (<code>action: skipped</code>).\n\n"
        "• <b>Задать вопрос:</b> Напишите вопрос со знаком вопроса <b>?</b> (например: <i>«Что делать со злостью?»</i>) или:\n"
        "  <code>/ask Как найти внутренний покой?</code>\n\n"
        "• <b>Векторный поиск:</b>\n"
        "  <code>/search медитация и спокойствие</code>\n\n"
        "• <b>Статистика базы знаний:</b>\n"
        "  <code>/stats</code>"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /ask <вопрос> — явный вопрос к базе знаний с RAG-генерацией.
    """
    if not context.args:
        await update.message.reply_text(
            "⚠️ Пожалуйста, укажите вопрос после команды.\n"
            "Пример: <code>/ask Как перестать тревожиться о будущем?</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    query = " ".join(context.args).strip()
    await handle_rag_question(update, context, query)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /stats — показывает состояние индекса Pinecone.
    """
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        stats = await asyncio.to_thread(pinecone_client.get_index_stats)
        namespaces = stats.get("namespaces", {})
        buddhism_info = namespaces.get(NAMESPACE, {})
        vector_count = buddhism_info.get("vector_count", 0)
        total_count = stats.get("total_vector_count", 0)
        dimension = stats.get("dimension", 1536)

        msg = (
            "📊 <b>Статус векторной базы Pinecone:</b>\n\n"
            f"• <b>Индекс:</b> <code>{INDEX_NAME}</code>\n"
            f"• <b>Раздел (namespace):</b> <code>{NAMESPACE}</code>\n"
            f"• <b>Фраз в разделе буддизма:</b> <code>{vector_count}</code>\n"
            f"• <b>Всего векторов в индексе:</b> <code>{total_count}</code>\n"
            f"• <b>Размерность векторов:</b> <code>{dimension}</code> dim"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}")
        await update.message.reply_text(f"❌ Ошибка получения статистики: {e}")


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /add <текст> — явное добавление фразы в базу.
    """
    if not context.args:
        await update.message.reply_text(
            "⚠️ Пожалуйста, укажите текст фразы после команды.\n"
            "Пример: <code>/add Спокойствие — величайшее проявление силы.</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    phrase_text = " ".join(context.args).strip()
    await process_save_phrase(update, phrase_text)


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /search <запрос> — семантический поиск без генерации LLM.
    """
    if not context.args:
        await update.message.reply_text(
            "⚠️ Укажите поисковый запрос.\n"
            "Пример: <code>/search как победить гнев</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    query = " ".join(context.args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        matches = await asyncio.to_thread(
            pinecone_client.search_by_text,
            query_text=query,
            top_k=3,
            namespace=NAMESPACE,
        )

        if not matches:
            await update.message.reply_text("🔍 Ничего не найдено в базе знаний.")
            return

        response_lines = [f"🔍 <b>Результаты поиска по запросу:</b> <i>«{query}»</i>\n"]
        for i, m in enumerate(matches, 1):
            score = m["score"] * 100
            meta = m.get("metadata", {})
            text = meta.get("text", "")
            author = meta.get("author", "Неизвестен")
            topic = meta.get("topic", "Мудрость")
            response_lines.append(
                f"<b>{i}. Сходство: {score:.1f}%</b> ({topic})\n"
                f"«{text}»\n"
                f"<i>— {author}</i>\n"
            )

        await update.message.reply_text("\n".join(response_lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}")
        await update.message.reply_text(f"❌ Ошибка при поиске: {e}")


# ------------------------------------------------------------------------------
# Обработка сообщений (Запоминание / RAG-ответы)
# ------------------------------------------------------------------------------


async def process_save_phrase(update: Update, phrase_text: str):
    """
    Логика векторизации и сохранения фразы в Pinecone с проверкой дедупликации.
    """
    chat_id = update.effective_chat.id
    user = update.effective_user
    author_name = user.full_name if user else "Пользователь"

    phrase_id = f"buddhism-user-{int(time.time())}"

    status_msg = await update.message.reply_text("⏳ <i>Проверяю дедупликацию и сохраняю в Pinecone...</i>", parse_mode=ParseMode.HTML)

    try:
        res = await asyncio.to_thread(
            pinecone_client.upsert_text,
            id=phrase_id,
            text=phrase_text,
            metadata={
                "author": author_name,
                "topic": "Пользовательская мудрость",
                "category": "user_wisdom",
                "user_id": user.id if user else None,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            namespace=NAMESPACE,
            filter_duplicates=True,
            on_duplicate="skip",
        )
    except Exception as e:
        logger.error(f"Ошибка сохранения фразы в Pinecone: {e}")
        try:
            await status_msg.edit_text(f"❌ Не удалось сохранить фразу: {e}")
        except Exception:
            await update.message.reply_text(f"❌ Не удалось сохранить фразу: {e}")
        return

    action = res.get("action", "created")

    if action == "skipped":
        dup = res.get("duplicate") or {}
        dup_meta = dup.get("metadata") or {}
        dup_text = dup.get("text") or dup_meta.get("text") or "..."
        dup_author = dup_meta.get("author") or "Неизвестен"
        dup_id = dup.get("id") or "..."
        dup_score = dup.get("score", 0.0) * 100
        threshold_pct = res.get("threshold", 0.88) * 100

        result_text = (
            "⚠️ <b>Похожая мысль уже есть в базе знаний! [action: skipped]</b>\n\n"
            f"📊 Косинусное сходство: <b>{dup_score:.1f}%</b> (порог: {threshold_pct:.0f}%)\n"
            f"📝 <b>Существующая запись:</b> «{dup_text}»\n"
            f"👤 <b>Автор:</b> {dup_author}\n"
            f"🔑 <b>ID:</b> <code>{dup_id}</code>\n\n"
            "🛡️ <i>Запись автоматически отфильтрована во избежание дублирования данных.</i>"
        )
    elif action == "updated":
        dup = res.get("duplicate") or {}
        dup_score = dup.get("score", 0.0) * 100
        result_text = (
            "🔄 <b>Запись успешно обновлена! [action: updated]</b>\n\n"
            f"📊 Обнаружено сходство: <b>{dup_score:.1f}%</b>\n"
            f"📝 <b>Обновленный текст:</b> «{phrase_text}»\n"
            f"👤 <b>Автор:</b> {author_name}\n"
            f"🔑 <b>ID:</b> <code>{res.get('id', phrase_id)}</code>"
        )
    else:
        result_text = (
            "✅ <b>Фраза успешно сохранена в базу знаний! [action: created]</b>\n\n"
            f"📝 <b>Текст:</b> «{phrase_text}»\n"
            f"👤 <b>Автор:</b> {author_name}\n"
            f"🔑 <b>ID:</b> <code>{phrase_id}</code>\n\n"
            "✨ Теперь эта мысль участвует в поиске и будет использоваться в ответах на вопросы!"
        )

    # Безопасное обновление статуса с повторной попыткой при разрыве соединения Telegram
    try:
        await status_msg.edit_text(result_text, parse_mode=ParseMode.HTML)
    except Exception as edit_err:
        logger.warning(f"Не удалось обновить статусное сообщение ({edit_err}), отправляю новое...")
        try:
            await update.message.reply_text(result_text, parse_mode=ParseMode.HTML)
        except Exception as send_err:
            logger.error(f"Не удалось отправить уведомление пользователю: {send_err}")



def generate_rag_answer_sync(query: str) -> tuple[str, list]:
    """
    Синхронный RAG: поиск совпадений в Pinecone и генерация ответа через OpenRouter/OpenAI.
    """
    matches = pinecone_client.search_by_text(
        query_text=query,
        top_k=3,
        namespace=NAMESPACE,
    )

    if not matches:
        return "В базе знаний пока нет подходящих учений по этой теме.", []

    context_chunks = []
    for i, m in enumerate(matches, 1):
        meta = m.get("metadata", {})
        context_chunks.append(
            f"{i}. «{meta.get('text')}» (Тема: {meta.get('topic')}, Автор: {meta.get('author')})"
        )
    context_text = "\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — мудрый и сострадательный наставник по буддийской философии и осознанности. "
                "Твоя цель — дать глубокий, теплый и практичный ответ на вопрос человека. "
                "Обязательно опирайся на приведенные ниже учения и цитаты из базы знаний, "
                "поясняя их смысл простым и живым языком. "
                "Используй форматирование с абзацами для удобного чтения в Telegram."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Вопрос человека: {query}\n\n"
                f"Мудрость из базы знаний:\n{context_text}\n\n"
                "Сформулируй свой ответ наставника."
            ),
        },
    ]

    response = pinecone_client.embedding_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )
    answer = response.choices[0].message.content
    return answer, matches


async def handle_rag_question(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    """
    Обработка вопроса: поиск в базе знаний и генерация ответа наставника.
    """
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        answer, matches = await asyncio.to_thread(generate_rag_answer_sync, query)

        # Формируем красивый блок с источниками
        sources_text = ""
        if matches:
            sources_lines = ["\n\n📜 <b>Опора на учения из базы:</b>"]
            for m in matches:
                meta = m.get("metadata", {})
                score = m.get("score", 0) * 100
                sources_lines.append(
                    f"• <i>«{meta.get('text')}»</i> — {meta.get('author', 'Буддизм')} ({score:.0f}%)"
                )
            sources_text = "\n".join(sources_lines)

        full_message = f"{answer}{sources_text}"
        await update.message.reply_text(full_message, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        try:
            await update.message.reply_text(f"🙏 {answer}")
        except Exception:
            await update.message.reply_text(f"❌ Произошла ошибка при формировании ответа: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Основной обработчик текстовых сообщений:
    1. Если триггер сохранения («Запомни: ...») -> сохраняет в Pinecone.
    2. Если вопрос ('?' или вопросительные слова) -> генерация RAG-ответа.
    3. Обычное сообщение пользователя (мысль/цитата) -> сохранение в Pinecone с автоматической дедупликацией!
    """
    text = update.message.text.strip()
    if not text:
        return

    # Проверяем явные триггеры сохранения
    save_text = extract_save_text(text)
    if save_text:
        await process_save_phrase(update, save_text)
        return

    # Если это вопрос — отвечаем через RAG
    if is_question_message(text):
        await handle_rag_question(update, context, text)
        return

    # Во всех остальных случаях обычное сообщение пользователя направляется
    # на векторизацию и сохранение в базу знаний с дедупликацией (action: created / action: skipped)
    await process_save_phrase(update, text)


# ------------------------------------------------------------------------------
# Запуск бота
# ------------------------------------------------------------------------------


def main():
    logger.info("Запуск Telegram-бота «Лотос Мудрости»...")
    app = ApplicationBuilder().token(TELEGRAM_KEY).build()

    # Регистрируем команды
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("add", add_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("search", search_command))

    # Регистрируем обработчик текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот успешно запущен и ожидает сообщений!")
    app.run_polling()


if __name__ == "__main__":
    main()
