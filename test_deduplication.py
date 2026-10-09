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
    test_ns = f"test_dedup_{int(time.time())}"

    client = PineconeManager(index_name=index_name, duplicate_threshold=0.88)
    print(f"Подключено к индексу '{index_name}', временный раздел '{test_ns}', порог: {client.duplicate_threshold}")

    ts = int(time.time())
    base_phrase = f"Истинный покой обретается в осознанности и тишине сердца {ts}"
    similar_phrase = f"Истинный покой обретается в осознанности и тишине сердца! {ts}"
    updated_phrase = f"Истинный покой обретается в осознанности и глубокой тишине сердца {ts}"

    print(f"\n1. Добавление первой фразы (ожидаем action: created)...")
    res1 = client.upsert_text(
        id=f"test-orig-{ts}",
        text=base_phrase,
        namespace=test_ns,
        filter_duplicates=True,
    )
    print(f" Результат 1: action = {res1.get('action')}, id = {res1.get('id')}")
    assert res1.get("action") == "created", f"Ожидалось action: created, получено {res1.get('action')}"
    print(" [OK] action: created подтверждено!")

    # Небольшая пауза для индексации вектора в Pinecone
    time.sleep(1.5)

    print(f"\n2. Добавление похожей фразы с on_duplicate='skip' (ожидаем action: skipped)...")
    res2 = client.upsert_text(
        id=f"test-skip-{ts}",
        text=similar_phrase,
        namespace=test_ns,
        filter_duplicates=True,
        on_duplicate="skip",
    )
    print(f" Результат 2: action = {res2.get('action')}, is_duplicate = {res2.get('is_duplicate')}")
    assert res2.get("action") == "skipped", f"Ожидалось action: skipped, получено {res2.get('action')}"
    assert res2.get("is_duplicate") is True, "Ожидалось is_duplicate == True"
    print(" [OK] action: skipped подтверждено!")

    print(f"\n3. Добавление похожей фразы с on_duplicate='update' (ожидаем action: updated)...")
    res3 = client.upsert_text(
        id=f"test-update-{ts}",
        text=updated_phrase,
        namespace=test_ns,
        filter_duplicates=True,
        on_duplicate="update",
    )
    print(f" Результат 3: action = {res3.get('action')}, is_duplicate = {res3.get('is_duplicate')}")
    assert res3.get("action") == "updated", f"Ожидалось action: updated, получено {res3.get('action')}"
    assert res3.get("is_duplicate") is True, "Ожидалось is_duplicate == True"
    print(" [OK] action: updated подтверждено!")

    print("\n4. Проверка пакетного сохранения с внутренней дедупликацией (upsert_texts)...")
    batch_items = [
        {"id": f"b1-{ts}", "text": f"Ум — это всё. Чем ты думаешь, тем ты и становишься {ts}."},
        {"id": f"b2-{ts}", "text": f"Ум — это абсолютно всё. Чем ты думаешь, тем ты и становишься {ts}."},
    ]
    batch_res = client.upsert_texts(
        items=batch_items,
        namespace=test_ns,
        filter_duplicates=True,
        duplicate_threshold=0.88,
    )
    print(f"Результат пакета: записано: {batch_res['upserted_count']}, пропущено дубликатов: {batch_res['skipped_duplicates_count']}")
    assert batch_res["skipped_duplicates_count"] >= 1, "Ожидалось отсечение дубликата в пакете"
    print(" [OK] Внутренняя дедупликация пакета сработала успешно!")

    # Очистим временный тестовый неймспейс
    client.delete_vectors(delete_all=True, namespace=test_ns)
    print(f" [OK] Временный тестовый раздел '{test_ns}' очищен.")


def test_verify_log_file():
    print("\n--- Тест 5: Проверка наличия записей в файле deduplication.log ---")
    log_file = os.getenv("LOG_FILE", "deduplication.log")
    assert os.path.exists(log_file), f"Файл логов {log_file} не найден!"

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    has_created = "action: created" in content
    has_skipped = "action: skipped" in content
    has_updated = "action: updated" in content

    print(f" Найдено 'action: created': {has_created}")
    print(f" Найдено 'action: skipped': {has_skipped}")
    print(f" Найдено 'action: updated': {has_updated}")

    assert has_created, "Лог не содержит записей 'action: created'"
    assert has_skipped, "Лог не содержит записей 'action: skipped'"
    assert has_updated, "Лог не содержит записей 'action: updated'"
    print(f" [OK] Файл {log_file} содержит все требуемые результаты логирования!")


if __name__ == "__main__":
    test_cosine_similarity_math()
    test_alias_and_defaults()
    test_filter_duplicate_results()
    test_live_pinecone_deduplication()
    test_verify_log_file()
    print("\n==============================================")
    print(" ВСЕ ТЕСТЫ ДЕДУПЛИКАЦИИ УСПЕШНО ПРОЙДЕНЫ!")
    print("==============================================\n")

