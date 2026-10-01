"""
Интерактивный консольный поиск и RAG-ответы по базе цитат буддизма в Pinecone.

Запуск:
    python chat_buddhism.py
"""

import sys
from pinecone_client import PineconeVectorClient

def search_quotes(client: PineconeVectorClient, query: str, top_k: int = 3):
    """
    Поиск наиболее подходящих цитат в Pinecone по смыслу вопроса.
    """
    results = client.search_by_text(
        query_text=query,
        top_k=top_k,
        namespace="buddhism"
    )
    return results

def ask_rag(client: PineconeVectorClient, query: str, top_k: int = 3, model: str = "openai/gpt-4o-mini") -> str:
    """
    RAG-ответ: находит цитаты в Pinecone и просит LLM через OpenRouter
    сформулировать мудрый ответ на основе найденной философии.
    """
    # 1. Поиск релевантных цитат в Pinecone
    matches = search_quotes(client, query, top_k=top_k)
    
    if not matches:
        return "В базе не найдено подходящих цитат."

    # Собираем контекст из найденных цитат
    context_chunks = []
    for i, m in enumerate(matches, 1):
        meta = m.get("metadata", {})
        context_chunks.append(
            f"{i}. «{meta.get('text')}» (Тема: {meta.get('topic')}, Источник: {meta.get('author')})"
        )
    context_text = "\n".join(context_chunks)

    # 2. Формируем запрос к языковой модели через OpenRouter
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — мудрый наставник по буддийской философии. "
                "Отвечай на вопрос собеседника спокойно, сострадательно и глубоко. "
                "Опирайся на приведенные ниже цитаты из базы знаний, поясняя их практический смысл для жизни человека."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Вопрос пользователя: {query}\n\n"
                f"Найденные учения и цитаты из базы знаний:\n{context_text}\n\n"
                "Дай ответ, используя эти учения."
            ),
        },
    ]

    try:
        response = client.embedding_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Ошибка при генерации ответа LLM: {e})"

def main():
    print("=" * 60)
    print("  ЛОТОС МУДРОСТИ: Поиск по векторизованным учениям буддизма")
    print("=" * 60)
    print("Подключение к Pinecone (индекс: test2, namespace: buddhism)...")
    
    client = PineconeVectorClient(index_name="test2")
    
    print("\nГотово! Вы можете задавать любые жизненные или философские вопросы.")
    print("Примеры вопросов:")
    print("  - Как перестать переживать из-за будущего?")
    print("  - Что делать, если на работе все раздражают?")
    print("  - Почему трудно расставаться с вещами и людьми?")
    print("  - Как научиться прощать?")
    print("\nДля выхода введите: 'exit' или 'выход'\n" + "-" * 60)

    SAVE_TRIGGERS = ["запомни", "запиши", "добавь", "сохрани", "зафиксируй"]

    while True:
        try:
            query = input("\n Введите вопрос или фразу: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "выход", "q"]:
                print("Мир вашему уму! До встречи.")
                break

            # Проверяем, не является ли это командой добавления фразы
            lower_query = query.lower()
            triggered_prefix = None
            for trigger in SAVE_TRIGGERS:
                if lower_query.startswith(trigger):
                    triggered_prefix = trigger
                    break

            if triggered_prefix:
                # Извлекаем текст после триггера и двоеточия/пробелов
                phrase_text = query[len(triggered_prefix):].strip()
                if phrase_text.startswith(":") or phrase_text.startswith("-"):
                    phrase_text = phrase_text[1:].strip()

                if not phrase_text:
                    print(" Текст для сохранения пуст. Пример: «Запомни: Спокойствие рождается изнутри»")
                    continue

                import time
                phrase_id = f"buddhism-user-{int(time.time())}"
                print(f" Сохраняю новую мысль в Pinecone (ID: {phrase_id})...")
                client.upsert_text(
                    id=phrase_id,
                    text=phrase_text,
                    metadata={"author": "Пользователь", "topic": "Пользовательская мысль", "category": "user_wisdom"},
                    namespace="buddhism"
                )
                print(f" Мысль успешно сохранена в векторизованную базу знаний!\nТекст: «{phrase_text}»")
                print("-" * 60)
                continue

            print("\n Ищу близкие по смыслу цитаты в Pinecone...")
            matches = search_quotes(client, query, top_k=3)

            print("\n--- Найденные совпадения в векторной базе ---")
            for i, m in enumerate(matches, 1):
                score = m["score"]
                meta = m.get("metadata", {})
                print(f"[{i}] Сходство: {score * 100:.1f}% | Тема: {meta.get('topic')}")
                print(f"    «{meta.get('text')}»")
                print(f"    — {meta.get('author')}\n")

            # Генерация ответа
            print(" Формирую осмысленный ответ наставника...")
            ai_answer = ask_rag(client, query, top_k=3)
            print("\n Ответ наставника:")
            print(ai_answer)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nВыход.")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
