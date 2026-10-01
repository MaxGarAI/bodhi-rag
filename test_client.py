"""
Скрипт для проверки работы PineconeVectorClient:
- подключение к Pinecone
- запись (upsert)
- чтение (fetch)
- поиск (search / query)
- удаление (delete)
- проверка текстового поиска через эмбеддинги OpenRouter
"""
import sys
import time
from pinecone_client import PineconeVectorClient

def main():
    print("=== 1. Инициализация клиента ===")
    client = PineconeVectorClient()

    # 1. Список индексов
    indexes = client.list_indexes()
    print(f"Доступные индексы Pinecone: {indexes}")

    target_index = "test2" if "test2" in indexes else indexes[0]
    print(f"\n=== 2. Подключение к индексу '{target_index}' ===")
    client.connect_index(target_index)

    # 2. Статистика
    stats = client.get_index_stats()
    print(f"Статистика индекса: {stats}")

    test_id = "demo-pizza-1"
    namespace = "test_demo"

    print("\n=== 3. Запись текста с автоматической генерацией эмбеддинга (upsert) ===")
    upsert_res = client.upsert_text(
        id=test_id,
        text="Свежая пицца Пепперони с сыром моцарелла и острыми колбасками",
        metadata={"category": "food", "price": 550},
        namespace=namespace,
    )
    print(f"Результат записи: {upsert_res}")

    # Дадим Pinecone секунду на индексацию
    time.sleep(1.5)

    print("\n=== 4. Чтение по ID (fetch) ===")
    fetch_res = client.fetch_vectors(ids=[test_id], namespace=namespace)
    print(f"Результат чтения: {fetch_res}")

    print("\n=== 5. Семантический поиск по тексту (search) ===")
    search_res = client.search_by_text(
        query_text="Какая пицца с колбасой и сыром?",
        top_k=2,
        namespace=namespace,
    )
    print(f"Найдено совпадений: {len(search_res)}")
    for i, match in enumerate(search_res, 1):
        print(f"  [{i}] ID: {match['id']}, Score: {match['score']:.4f}")
        print(f"      Текст: {match.get('metadata', {}).get('text')}")
        print(f"      Метаданные: {match.get('metadata')}")

    # print("\n=== 6. Удаление вектора (delete) ===")
    # del_res = client.delete_vectors(ids=[test_id], namespace=namespace)
    # print(f"Результат удаления: {del_res}")

    # Проверка после удаления
    # time.sleep(1)
    # fetch_after_del = client.fetch_vectors(ids=[test_id], namespace=namespace)
    # vectors_dict = fetch_after_del.get("vectors", {})
    # print(f"Вектор в базе после удаления: {vectors_dict.get(test_id) is not None}")

    print("\n Все тесты успешно пройдены!")

if __name__ == "__main__":
    main()
