"""
Тестирование логики косинусного сходства (Cosine Similarity)
и автоматической фильтрации дубликатов в PineconeManager.
"""

import math
import os
import time
from dotenv import load_dotenv

from pinecone_client import (
    PineconeManager,
    PineconeVectorClient,
    compute_cosine_similarity,
    DEFAULT_DUPLICATE_THRESHOLD,
)

load_dotenv()


def test_cosine_similarity_math():
    print("\n--- Тест 1: Математическая корректность косинусного сходства ---")

    # Идентичные векторы
    sim_identical = compute_cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(sim_identical - 1.0) < 1e-6, f"Ожидалось 1.0, получено {sim_identical}"
    print(f" [OK] Идентичные векторы: similarity = {sim_identical:.4f} (ожидалось 1.0)")

    # Противоположные векторы
    sim_opposite = compute_cosine_similarity([1.0, 0.0], [-1.0, 0.0])
    assert abs(sim_opposite - (-1.0)) < 1e-6, f"Ожидалось -1.0, получено {sim_opposite}"
    print(f" [OK] Противоположные векторы: similarity = {sim_opposite:.4f} (ожидалось -1.0)")

    # Ортогональные векторы
    sim_ortho = compute_cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(sim_ortho - 0.0) < 1e-6, f"Ожидалось 0.0, получено {sim_ortho}"
    print(f" [OK] Ортогональные векторы: similarity = {sim_ortho:.4f} (ожидалось 0.0)")

    # Близкие векторы (угол ~11 градусов -> cos ~ 0.98)
    sim_close = compute_cosine_similarity([1.0, 0.1, 0.0], [1.0, 0.3, 0.0])
    assert sim_close > 0.95, f"Ожидалось > 0.95, получено {sim_close}"
    print(f" [OK] Близкие векторы: similarity = {sim_close:.4f} (> 0.95)")

    # Нулевой вектор
    sim_zero = compute_cosine_similarity([0.0, 0.0], [1.0, 2.0])
    assert sim_zero == 0.0
    print(f" [OK] Нулевой вектор: similarity = {sim_zero:.4f} (0.0)")


def test_alias_and_defaults():
    print("\n--- Тест 2: Проверка алиаса PineconeManager и порога по умолчанию ---")
    assert PineconeManager is PineconeVectorClient, "PineconeManager должен быть алиасом PineconeVectorClient"
    print(" [OK] PineconeManager is PineconeVectorClient")

    assert DEFAULT_DUPLICATE_THRESHOLD == 0.88, "Порог по умолчанию должен быть 0.88"
    print(f" [OK] DEFAULT_DUPLICATE_THRESHOLD = {DEFAULT_DUPLICATE_THRESHOLD}")


def test_filter_duplicate_results():
    print("\n--- Тест 3: Фильтрация дубликатов в результатах поиска ---")
    client = PineconeManager()

    matches = [
        {"id": "doc1", "score": 0.95, "values": [1.0, 0.0, 0.0], "metadata": {"text": "Документ 1"}},
        {"id": "doc2", "score": 0.94, "values": [0.999, 0.01, 0.0], "metadata": {"text": "Почти Документ 1"}},
        {"id": "doc3", "score": 0.80, "values": [0.0, 1.0, 0.0], "metadata": {"text": "Документ 2 (другой)"}},
    ]

    filtered = client.filter_duplicate_results(matches, threshold=0.90)
    assert len(filtered) == 2, f"Ожидалось 2 уникальных документа, получено {len(filtered)}"
    assert filtered[0]["id"] == "doc1"
    assert filtered[1]["id"] == "doc3"
    print(f" [OK] Исходно {len(matches)} результатов -> Отфильтровано {len(filtered)} уникальных:")
    for f in filtered:
        print(f"      - {f['id']}: {f['metadata']['text']}")


def test_live_pinecone_deduplication():
    print("\n--- Тест 4: Проверка дедупликации в реальном индексе Pinecone ---")
    index_name = os.getenv("PINECONE_INDEX", "test2")
    namespace = "buddhism"

    client = PineconeManager(index_name=index_name, duplicate_threshold=0.88)
    print(f"Подключено к индексу '{index_name}', раздел '{namespace}', порог: {client.duplicate_threshold}")

    # Известная фраза, которая уже есть в базе знаний буддизма
    existing_phrase = "Спокойствие приходит изнутри. Не ищи его снаружи."

    print(f"\nПопытка добавить фразу-дубликат: «{existing_phrase}»...")
    res_dup = client.upsert_text(
        id=f"test-dup-{int(time.time())}",
        text=existing_phrase,
        namespace=namespace,
        filter_duplicates=True,
    )

    print(f"Результат проверки: is_duplicate = {res_dup.get('is_duplicate')}")
    if res_dup.get("is_duplicate"):
        dup_info = res_dup.get("duplicate", {})
        score = dup_info.get("score", 0.0)
        print(f" [OK] Дубликат успешно обнаружен! Сходство: {score * 100:.2f}% (порог: {res_dup.get('threshold') * 100:.0f}%)")
        print(f"      Совпал с существующей записью ID: '{dup_info.get('id')}'")
        print(f"      Текст в базе: «{dup_info.get('text')}»")
    else:
        print(" [INFO] Точного дубликата в базе не нашлось (возможно, база еще не заполнена этими фразами).")

    print("\nПроверка пакетного сохранения с внутренней дедупликацией (upsert_texts)...")
    batch_items = [
        {"id": "b1", "text": "Ум — это всё. Чем ты думаешь, тем ты и становишься."},
        {"id": "b2", "text": "Ум — это абсолютно всё. О чем ты думаешь, тем ты и становишься."},  # дубликат b1
    ]
    batch_res = client.upsert_texts(
        items=batch_items,
        namespace="test_dedup_tmp",
        filter_duplicates=True,
        duplicate_threshold=0.88,
    )
    print(f"Результат пакета: записано: {batch_res['upserted_count']}, пропущено дубликатов: {batch_res['skipped_duplicates_count']}")
    assert batch_res["skipped_duplicates_count"] >= 1, "Ожидалось отсечение хотя бы одного дубликата в пакете"
    print(" [OK] Внутренняя дедупликация пакета сработала успешно!")

    # Очистим временный неймспейс
    client.delete_vectors(delete_all=True, namespace="test_dedup_tmp")
    print(" [OK] Временный тестовый раздел очищен.")


if __name__ == "__main__":
    test_cosine_similarity_math()
    test_alias_and_defaults()
    test_filter_duplicate_results()
    test_live_pinecone_deduplication()
    print("\n==============================================")
    print(" ВСЕ ТЕСТЫ ДЕДУПЛИКАЦИИ УСПЕШНО ПРОЙДЕНЫ!")
    print("==============================================\n")
