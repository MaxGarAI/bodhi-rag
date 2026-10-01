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

from pinecone_client import PineconeVectorClient

# Загружаем переменные из .env
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Инициализация переменных окружения
TELEGRAM_KEY = os.getenv("TELEGRAM_KEY") or os.getenv("TELEGRAM_BOT_TOKEN")
INDEX_NAME = os.getenv("PINECONE_INDEX", "test2")
NAMESPACE = "buddhism"

if not TELEGRAM_KEY:
    raise ValueError("TELEGRAM_KEY не найден в файле .env!")

# Инициализация клиента Pinecone
pinecone_client = PineconeVectorClient(index_name=INDEX_NAME)

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
        "1. <b>Отвечать на вопросы</b> — просто напишите любой вопрос (например: <i>«Как справиться с тревогой?»</i> или <i>«Что делать со злостью?»</i>).\n"
        "2. <b>Запоминать новые фразы</b> — напишите:\n"
        "   • <code>Запомни: Спокойствие рождается изнутри</code>\n"
        "   • <code>Запиши: Каждое утро мы рождаемся заново</code>\n"
        "   • или команду <code>/add Ваш текст</code>\n\n"
        "<b>Команды:</b>\n"
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
        "• <b>Задать вопрос:</b> Просто напишите мне то, что вас волнует. "
        "Я найду близкие по смыслу учения в Pinecone и сформулирую ответ.\n\n"
        "• <b>Добавить мудрость:</b> Начните фразу со слов <i>«Запомни»</i>, <i>«Запиши»</i>, <i>«Добавь»</i> или используйте:\n"
        "  <code>/add Не привязывайся к результату</code>\n\n"
        "• <b>Векторный поиск:</b>\n"
        "  <code>/search медитация и спокойствие</code>\n\n"
        "• <b>Статистика:</b>\n"
        "  <code>/stats</code>"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


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
    Логика векторизации и сохранения фразы в Pinecone.
    """
    chat_id = update.effective_chat.id
    user = update.effective_user
    author_name = user.full_name if user else "Пользователь"

    phrase_id = f"buddhism-user-{int(time.time())}"

    status_msg = await update.message.reply_text("⏳ <i>Векторизую и сохраняю в Pinecone...</i>", parse_mode=ParseMode.HTML)

    try:
        await asyncio.to_thread(
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
        )

        success_text = (
            "✅ <b>Фраза успешно сохранена в базу знаний!</b>\n\n"
            f"📝 <b>Текст:</b> «{phrase_text}»\n"
            f"👤 <b>Автор:</b> {author_name}\n"
            f"🔑 <b>ID:</b> <code>{phrase_id}</code>\n\n"
            "✨ Теперь эта мысль участвует в поиске и будет использоваться в ответах на вопросы!"
        )
        await status_msg.edit_text(success_text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Ошибка сохранения фразы: {e}")
        await status_msg.edit_text(f"❌ Не удалось сохранить фразу: {e}")


def generate_rag_answer_sync(query: str) -> tuple[str, list]:
    """
    Синхронный RAG: поиск совпадений в Pinecone и генерация ответа через OpenRouter.
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Основной обработчик текстовых сообщений:
    1. Если триггер сохранения («Запомни: ...») -> сохраняет в Pinecone.
    2. Иначе -> выполняет семантический поиск и генерирует RAG-ответ.
    """
    text = update.message.text.strip()
    if not text:
        return

    # Проверяем на триггеры сохранения
    save_text = extract_save_text(text)
    if save_text:
        await process_save_phrase(update, save_text)
        return

    # Во всех остальных случаях — это вопрос к базе
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        answer, matches = await asyncio.to_thread(generate_rag_answer_sync, text)

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
        # Если Telegram отклонил HTML-разметку из-за спецсимволов, пробуем отправить без Markdown/HTML
        try:
            await update.message.reply_text(f"🙏 {answer}")
        except Exception:
            await update.message.reply_text(f"❌ Произошла ошибка при формировании ответа: {e}")


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
    app.add_handler(CommandHandler("search", search_command))

    # Регистрируем обработчик текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот успешно запущен и ожидает сообщений!")
    app.run_polling()


if __name__ == "__main__":
    main()
