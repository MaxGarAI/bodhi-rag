import os
import time
import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка базового логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Порог косинусного сходства по умолчанию для автоматической фильтрации дубликатов
DEFAULT_DUPLICATE_THRESHOLD: float = 0.88


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Вычисление косинусного сходства (Cosine Similarity) между двумя векторами.

    Формула:
        cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

    Args:
        vec1: Первый числовой вектор.
        vec2: Второй числовой вектор.

    Returns:
        float: Значение косинусного сходства в диапазоне [-1.0, 1.0] (для нормализованных
               эмбеддингов обычно [0.0, 1.0]). 1.0 означает полную сонаправленность.

    Raises:
        ValueError: Если векторы имеют разную размерность.
    """
    if not vec1 or not vec2:
        return 0.0
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Размерности векторов не совпадают: {len(vec1)} != {len(vec2)}"
        )

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    similarity = dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(-1.0, min(1.0, float(similarity)))


class PineconeVectorClient:
    """
    Клиент для взаимодействия с векторной базой данных Pinecone (PineconeManager)
    с поддержкой генерации эмбеддингов через OpenRouter и автоматической
    фильтрации дубликатов на основе косинусного сходства (Cosine Similarity).

    Предоставляет методы для:
      - подключения к базе и индексам Pinecone,
      - записи (upsert) векторов и текстов с дедупликацией,
      - чтения (fetch) векторов по ID,
      - семантического поиска (query / search) векторов и текстовых запросов,
      - проверки наличия дубликатов (find_duplicate),
      - удаления (delete) векторов по ID, фильтрам или полной очистки namespace.
    """

    compute_cosine_similarity = staticmethod(compute_cosine_similarity)

    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        default_embedding_model: str = "openai/text-embedding-3-small",
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    ):
        """
        Инициализация клиента Pinecone и OpenRouter.

        Args:
            pinecone_api_key (Optional[str]): API-ключ Pinecone. Если не указан,
                считывается из переменной окружения PINECONE_KEY или PINECONE_API_KEY.
            index_name (Optional[str]): Имя индекса для автоматического подключения.
                Если не указано, подключение можно выполнить позже вызовом connect_index().
            openrouter_api_key (Optional[str]): API-ключ OpenRouter для генерации эмбеддингов.
                Если не указан, считывается из переменной окружения OPENROUTER_API_KEY.
            default_embedding_model (str): Название модели эмбеддингов в OpenRouter по умолчанию.
                По умолчанию: "openai/text-embedding-3-small" (размерность 1536).
            duplicate_threshold (float): Порог косинусного сходства для отсечения дубликатов.
                По умолчанию 0.88 или считывается из DUPLICATE_THRESHOLD / SIMILARITY_THRESHOLD.

        Raises:
            ValueError: Если ключ Pinecone не найден ни в аргументах, ни в .env.
        """
        self.pinecone_api_key = (
            pinecone_api_key
            or os.getenv("PINECONE_KEY")
            or os.getenv("PINECONE_API_KEY")
        )
        if not self.pinecone_api_key:
            raise ValueError(
                "API-ключ Pinecone не найден. Передайте pinecone_api_key "
                "или установите переменную PINECONE_KEY в файле .env"
            )

        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.default_embedding_model = default_embedding_model

        # Установка порога косинусного сходства для фильтрации дубликатов
        env_threshold = os.getenv("DUPLICATE_THRESHOLD") or os.getenv("SIMILARITY_THRESHOLD")
        try:
            self.duplicate_threshold = (
                float(env_threshold) if env_threshold is not None else duplicate_threshold
            )
        except ValueError:
            self.duplicate_threshold = duplicate_threshold

        # Инициализация клиента Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = None
        self.current_index_name: Optional[str] = None

        # Инициализация клиента OpenRouter (для генерации эмбеддингов)
        self.embedding_client: Optional[OpenAI] = None
        if self.openrouter_api_key:
            self.embedding_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Bot_RAG",
                },
            )
        else:
            logger.warning(
                "OPENROUTER_API_KEY не задан. Функции генерации эмбеддингов будут недоступны."
            )

        if index_name:
            self.connect_index(index_name)

    # --------------------------------------------------------------------------
    # Подключение и управление индексами
    # --------------------------------------------------------------------------

    def list_indexes(self, retries: int = 3, delay: float = 1.0) -> List[str]:
        """
        Получение списка названий всех доступных индексов в аккаунте Pinecone с повторными попытками.

        Args:
            retries (int): Количество попыток при сбоях сети (по умолчанию 3).
            delay (float): Задержка в секундах между попытками (по умолчанию 1.0).

        Returns:
            List[str]: Список имен индексов (например, ['rag-demo-index', 'test2']).

        Raises:
            Exception: При ошибке обращения к API Pinecone после всех попыток.
        """
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                indexes = self.pc.list_indexes()
                return [idx.name for idx in indexes]
            except Exception as e:
                last_error = e
                if attempt < retries:
                    logger.warning(
                        f"Временный сбой сети при получении индексов Pinecone (попытка {attempt}/{retries}): {e}. "
                        f"Повтор через {delay:.1f}с..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Ошибка при получении списка индексов после {retries} попыток: {e}")
        raise last_error

    def connect_index(self, index_name: str) -> Any:
        """
        Подключение к существующему индексу Pinecone по его названию.
        Включает валидацию наличия индекса и устойчивость к кратковременным сетевым сбоям.

        Args:
            index_name (str): Название индекса Pinecone.

        Returns:
            Any: Экземпляр подключенного индекса Pinecone (Index).

        Raises:
            ValueError: Если индекс с таким именем отсутствует в аккаунте.
            Exception: При ошибке подключения к индексу.
        """
        try:
            # Пытаемся проверить наличие индекса в списке аккаунта
            try:
                available_indexes = self.list_indexes(retries=2, delay=1.0)
                if available_indexes and index_name not in available_indexes:
                    raise ValueError(
                        f"Индекс '{index_name}' не найден. Доступные индексы: {available_indexes}"
                    )
            except ValueError:
                raise
            except Exception as net_err:
                logger.warning(
                    f"Не удалось получить предварительный список индексов ({net_err}). "
                    f"Выполняется прямое подключение к индексу '{index_name}'..."
                )

            self.index = self.pc.Index(index_name)
            self.current_index_name = index_name
            logger.info(f"Успешно подключено к индексу Pinecone: '{index_name}'")
            return self.index
        except Exception as e:
            logger.error(f"Ошибка при подключении к индексу '{index_name}': {e}")
            raise

    def get_index_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение статистики текущего подключенного индекса.

        Args:
            namespace (Optional[str]): Ограничить статистику конкретным пространством имен (если требуется).

        Returns:
            Dict[str, Any]: Словарь со статистикой индекса:
                - 'dimension' (int): Размерность векторов.
                - 'total_vector_count' (int): Общее количество векторов.
                - 'metric' (str): Метрика расстояния (cosine, euclidean, dotproduct).
                - 'namespaces' (dict): Информация по неймспейсам и количеству векторов в них.

        Raises:
            RuntimeError: Если индекс еще не подключен через connect_index().
        """
        self._ensure_index_connected()
        try:
            stats = self.index.describe_index_stats()
            if hasattr(stats, "to_dict"):
                return stats.to_dict()
            return dict(stats)
        except Exception as e:
            logger.error(f"Ошибка при получении статистики индекса: {e}")
            raise

    # --------------------------------------------------------------------------
    # Запись векторов (Upsert)
    # --------------------------------------------------------------------------

    def upsert_vectors(
        self,
        vectors: List[Union[Dict[str, Any], tuple]],
        namespace: Optional[str] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Запись (вставка или обновление) списка векторов в индекс Pinecone.
        Поддерживает автоматическую разбивку на пакеты (батчи) заданного размера.

        Args:
            vectors (List[Union[Dict[str, Any], tuple]]): Список векторов.
                Каждый элемент может быть:
                  - словарем вида {"id": "vec1", "values": [0.1, ...], "metadata": {"key": "val"}}
                  - кортежем вида ("vec1", [0.1, ...], {"key": "val"})
            namespace (Optional[str]): Пространство имен (namespace) для изоляции данных.
                Если не указано, используется namespace по умолчанию ("").
            batch_size (int): Размер пачки векторов для одного запроса. По умолчанию 100.

        Returns:
            Dict[str, Any]: Словарь с результатами операции:
                - 'upserted_count' (int): Суммарное количество успешно записанных векторов.
                - 'batches_count' (int): Количество отправленных батчей.

        Raises:
            RuntimeError: Если индекс не подключен.
            ValueError: Если список векторов пуст или некорректен.
            Exception: При ошибке передачи данных в Pinecone.
        """
        self._ensure_index_connected()

        if not vectors:
            raise ValueError("Список vectors не может быть пустым.")

        total_upserted = 0
        total_batches = 0

        try:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                kwargs = {"vectors": batch}
                if namespace is not None:
                    kwargs["namespace"] = namespace

                response = self.index.upsert(**kwargs)
                count = getattr(response, "upserted_count", len(batch))
                total_upserted += count
                total_batches += 1

            logger.info(
                f"Успешно записано {total_upserted} векторов в индекс '{self.current_index_name}' "
                f"(пакетов: {total_batches}, namespace: '{namespace or ''}')."
            )
            return {
                "upserted_count": total_upserted,
                "batches_count": total_batches,
            }
        except Exception as e:
            logger.error(f"Ошибка при записи векторов: {e}")
            raise

    # --------------------------------------------------------------------------
    # Чтение векторов (Fetch)
    # --------------------------------------------------------------------------

    def fetch_vectors(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Чтение векторов и их метаданных из индекса Pinecone по списку их ID.

        Args:
            ids (List[str]): Список идентификаторов векторов для получения.
            namespace (Optional[str]): Пространство имен (namespace), в котором хранятся векторы.

        Returns:
            Dict[str, Any]: Словарь с найденными векторами:
                - 'vectors': Dict[str, dict] со структурой:
                    {
                        "id": str,
                        "values": List[float],
                        "metadata": Dict[str, Any]
                    }
                - 'namespace': str, использованный неймспейс.

        Raises:
            RuntimeError: Если индекс не подключен.
            ValueError: Если список ids пуст.
            Exception: При ошибке обращения к Pinecone.
        """
        self._ensure_index_connected()

        if not ids:
            raise ValueError("Список ids не может быть пустым.")

        try:
            kwargs = {"ids": ids}
            if namespace is not None:
                kwargs["namespace"] = namespace

            response = self.index.fetch(**kwargs)
            if hasattr(response, "to_dict"):
                return response.to_dict()
            return dict(response)
        except Exception as e:
            logger.error(f"Ошибка при чтении векторов по IDs {ids}: {e}")
            raise

    # --------------------------------------------------------------------------
    # Поиск векторов (Query / Search)
    # --------------------------------------------------------------------------

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
        namespace: Optional[str] = None,
        deduplicate: bool = False,
        duplicate_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Поиск наиболее похожих векторов в индексе Pinecone по вектору запроса.

        Args:
            query_vector (List[float]): Вектор запроса (список вещественных чисел).
            top_k (int): Максимальное количество возвращаемых наиболее похожих результатов. По умолчанию 5.
            filter (Optional[Dict[str, Any]]): Фильтр по метаданным (например, {"category": {"$eq": "pizza"}}).
            include_metadata (bool): Включать ли метаданные векторов в результат. По умолчанию True.
            include_values (bool): Включать ли числовые значения векторов в результат. По умолчанию False.
            namespace (Optional[str]): Пространство имен для поиска.
            deduplicate (bool): Исключать ли близкие дубликаты из результатов поиска (по умолчанию False).
            duplicate_threshold (Optional[float]): Порог косинусного сходства для отсечения дубликатов
                (по умолчанию self.duplicate_threshold, 0.88).

        Returns:
            List[Dict[str, Any]]: Список найденных совпадений, отсортированный по убыванию сходства.
                Каждый элемент содержит:
                - 'id' (str): Уникальный ID вектора.
                - 'score' (float): Оценка сходства (косинусное сходство / скалярное произведение).
                - 'metadata' (Optional[Dict[str, Any]]): Метаданные (если include_metadata=True).
                - 'values' (Optional[List[float]]): Векторные значения (если include_values=True).

        Raises:
            RuntimeError: Если индекс не подключен.
            ValueError: Если query_vector пуст.
            Exception: При ошибке выполнения запроса к Pinecone.
        """
        self._ensure_index_connected()

        if not query_vector:
            raise ValueError("query_vector не может быть пустым.")

        try:
            fetch_top_k = max(top_k * 2, top_k + 5) if deduplicate else top_k
            fetch_values = True if deduplicate else include_values

            kwargs = {
                "vector": query_vector,
                "top_k": fetch_top_k,
                "include_metadata": include_metadata,
                "include_values": fetch_values,
            }
            if filter is not None:
                kwargs["filter"] = filter
            if namespace is not None:
                kwargs["namespace"] = namespace

            response = self.index.query(**kwargs)
            matches = getattr(response, "matches", [])

            results = []
            for match in matches:
                item = {
                    "id": getattr(match, "id", None),
                    "score": getattr(match, "score", None),
                }
                if include_metadata:
                    item["metadata"] = getattr(match, "metadata", None)
                if fetch_values:
                    item["values"] = getattr(match, "values", None)
                results.append(item)

            if deduplicate:
                results = self.filter_duplicate_results(
                    results,
                    threshold=duplicate_threshold,
                )
                if not include_values:
                    for item in results:
                        item.pop("values", None)
                results = results[:top_k]

            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске векторов: {e}")
            raise

    # --------------------------------------------------------------------------
    # Дедупликация и косинусное сходство (Deduplication)
    # --------------------------------------------------------------------------

    def find_duplicate(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        namespace: Optional[str] = None,
        threshold: Optional[float] = None,
        model: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Проверка наличия дубликата в индексе Pinecone по косинусному сходству.

        Ищет наиболее похожий вектор в указанном namespace. Если косинусное сходство (score)
        превышает установленный порог (threshold), возвращает данные найденного дубликата.

        Args:
            text (Optional[str]): Исходный текст для поиска дубликата (будет векторизован).
            vector (Optional[List[float]]): Готовый вектор (если эмбеддинг уже вычислен).
            namespace (Optional[str]): Пространство имен Pinecone.
            threshold (Optional[float]): Порог косинусного сходства (по умолчанию self.duplicate_threshold, 0.88).
            model (Optional[str]): Модель эмбеддингов для векторизации text.
            filter (Optional[Dict[str, Any]]): Фильтр по метаданным.

        Returns:
            Optional[Dict[str, Any]]: Данные дубликата или None, если дубликат не найден:
                {
                    "id": str,
                    "score": float,
                    "text": str,
                    "metadata": Dict[str, Any]
                }
        """
        if vector is None:
            if not text:
                raise ValueError("Необходимо указать text или vector для поиска дубликата.")
            vector = self.get_embedding(text=text, model=model)

        min_threshold = threshold if threshold is not None else self.duplicate_threshold

        matches = self.search_vectors(
            query_vector=vector,
            top_k=1,
            filter=filter,
            include_metadata=True,
            include_values=False,
            namespace=namespace,
            deduplicate=False,
        )

        if matches:
            top_match = matches[0]
            score = top_match.get("score")
            if score is not None and score >= min_threshold:
                meta = top_match.get("metadata") or {}
                return {
                    "id": top_match.get("id"),
                    "score": score,
                    "text": meta.get("text", ""),
                    "metadata": meta,
                }

        return None

    def filter_duplicate_results(
        self,
        matches: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Фильтрация семантических дубликатов из списка результатов поиска.

        Попарно сравнивает результаты с помощью косинусного сходства (Cosine Similarity).
        Если последующий результат имеет косинусное сходство с уже добавленным результатом
        выше установленного порога, он отсекается как семантический дубликат.

        Args:
            matches (List[Dict[str, Any]]): Список результатов поиска.
            threshold (Optional[float]): Порог косинусного сходства (по умолчанию self.duplicate_threshold).

        Returns:
            List[Dict[str, Any]]: Список уникальных результатов поиска.
        """
        if not matches:
            return []

        limit_threshold = threshold if threshold is not None else self.duplicate_threshold
        unique_matches: List[Dict[str, Any]] = []

        # Проверяем наличие векторов в matches
        needs_embedding = any(not m.get("values") for m in matches)
        if needs_embedding and self.embedding_client:
            texts_to_embed = [m.get("metadata", {}).get("text", "") for m in matches]
            if any(texts_to_embed):
                try:
                    embeddings = self.get_embeddings(texts=texts_to_embed)
                    for m, emb in zip(matches, embeddings):
                        if not m.get("values"):
                            m["values"] = emb
                except Exception as e:
                    logger.warning(f"Не удалось сгенерировать эмбеддинги для дедупликации: {e}")

        for candidate in matches:
            cand_vec = candidate.get("values")
            if not cand_vec:
                unique_matches.append(candidate)
                continue

            is_duplicate = False
            for accepted in unique_matches:
                acc_vec = accepted.get("values")
                if acc_vec:
                    sim = compute_cosine_similarity(cand_vec, acc_vec)
                    if sim >= limit_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_matches.append(candidate)

        return unique_matches

    # --------------------------------------------------------------------------
    # Удаление векторов (Delete)
    # --------------------------------------------------------------------------

    def delete_vectors(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Удаление векторов из индекса Pinecone.
        Поддерживает удаление по списку ID, по фильтру метаданных или удаление всех векторов.

        Args:
            ids (Optional[List[str]]): Список ID векторов для удаления.
            delete_all (bool): Если True, удаляет ВСЕ векторы в указанном namespace. По умолчанию False.
            filter (Optional[Dict[str, Any]]): Фильтр метаданных для удаления подходящих векторов.
            namespace (Optional[str]): Пространство имен, из которого удаляются векторы.

        Returns:
            Dict[str, Any]: Словарь со статусом выполнения операции:
                {"status": "success", "deleted_ids": ids, "namespace": namespace, "delete_all": delete_all}

        Raises:
            RuntimeError: Если индекс не подключен.
            ValueError: Если не задан ни один из параметров удаления (ids, delete_all, filter).
            Exception: При ошибке удаления в Pinecone.
        """
        self._ensure_index_connected()

        if not ids and not delete_all and not filter:
            raise ValueError(
                "Для удаления необходимо указать хотя бы один критерий: "
                "список 'ids', 'filter' или флаг 'delete_all=True'."
            )

        try:
            kwargs = {}
            if ids:
                kwargs["ids"] = ids
            if delete_all:
                kwargs["delete_all"] = True
            if filter:
                kwargs["filter"] = filter
            if namespace is not None:
                kwargs["namespace"] = namespace

            self.index.delete(**kwargs)
            logger.info(
                f"Успешно удалены векторы из '{self.current_index_name}' "
                f"(ids={ids}, delete_all={delete_all}, namespace='{namespace or ''}')."
            )
            return {
                "status": "success",
                "deleted_ids": ids,
                "delete_all": delete_all,
                "namespace": namespace or "",
            }
        except Exception as e:
            logger.error(f"Ошибка при удалении векторов: {e}")
            raise

    # --------------------------------------------------------------------------
    # Интеграция с OpenRouter (Эмбеддинги и высокоуровневые RAG-методы)
    # --------------------------------------------------------------------------

    def get_embedding(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Генерация векторного представления (эмбеддинга) для одной текстовой строки через OpenRouter.

        Args:
            text (str): Исходный текст для векторизации.
            model (Optional[str]): Модель эмбеддингов в OpenRouter.
                Если не указана, используется self.default_embedding_model
                ("openai/text-embedding-3-small").

        Returns:
            List[float]: Вектор вещественных чисел (размерность 1536 для text-embedding-3-small).

        Raises:
            RuntimeError: Если OPENROUTER_API_KEY не был задан при инициализации.
            ValueError: Если передан пустой текст.
            Exception: При ошибке запроса к OpenRouter.
        """
        embeddings = self.get_embeddings(texts=[text], model=model)
        return embeddings[0]

    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Пакетная генерация векторных представлений (эмбеддингов) для списка текстов через OpenRouter.

        Args:
            texts (List[str]): Список строк для векторизации.
            model (Optional[str]): Модель эмбеддингов в OpenRouter.
                Если не указана, используется self.default_embedding_model.

        Returns:
            List[List[float]]: Список векторов вещественных чисел в том же порядке, что и входные тексты.

        Raises:
            RuntimeError: Если OPENROUTER_API_KEY не был задан при инициализации.
            ValueError: Если передан пустой список.
            Exception: При ошибке запроса к OpenRouter.
        """
        if not self.embedding_client:
            raise RuntimeError(
                "Клиент генерации эмбеддингов недоступен: OPENROUTER_API_KEY не настроен."
            )

        if not texts:
            raise ValueError("Список текстов texts не может быть пустым.")

        embedding_model = model or self.default_embedding_model

        try:
            response = self.embedding_client.embeddings.create(
                input=texts,
                model=embedding_model,
            )
            # Сортируем результаты по индексу на случай, если API вернет не по порядку
            sorted_data = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            logger.error(
                f"Ошибка при получении пакета эмбеддингов от OpenRouter ({embedding_model}): {e}"
            )
            raise

    def upsert_texts(
        self,
        items: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 50,
        filter_duplicates: bool = True,
        duplicate_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Пакетная запись текстов: вычисляет эмбеддинги для списка записей и
        сохраняет их в Pinecone с опциональной фильтрацией дубликатов по косинусному сходству.

        Args:
            items (List[Dict[str, Any]]): Список словарей с текстами.
                Каждый элемент должен содержать:
                  - "id" (str): Уникальный ID.
                  - "text" (str): Текст документа/фразы.
                  - "metadata" (Optional[Dict[str, Any]]): Дополнительные метаданные (опционально).
            namespace (Optional[str]): Пространство имен Pinecone.
            model (Optional[str]): Модель эмбеддингов OpenRouter.
            batch_size (int): Размер пакета для генерации эмбеддингов и записи. По умолчанию 50.
            filter_duplicates (bool): Автоматически фильтровать дубликаты перед сохранением (по умолчанию True).
            duplicate_threshold (Optional[float]): Порог косинусного сходства (по умолчанию self.duplicate_threshold, 0.88).

        Returns:
            Dict[str, Any]: Словарь с результатами:
                - 'upserted_count' (int): Общее число сохраненных векторов.
                - 'skipped_duplicates_count' (int): Число пропущенных дубликатов.
                - 'duplicates' (List[Dict[str, Any]]): Список пропущенных дубликатов.
                - 'batches_count' (int): Число отправленных пакетов.
                - 'threshold' (float): Порог косинусного сходства.

        Raises:
            ValueError: Если items пуст или структура элементов некорректна.
        """
        if not items:
            raise ValueError("Список items не может быть пустым.")

        threshold = (
            duplicate_threshold
            if duplicate_threshold is not None
            else self.duplicate_threshold
        )

        total_upserted = 0
        total_batches = 0
        skipped_duplicates: List[Dict[str, Any]] = []

        accepted_records: List[Dict[str, Any]] = []
        accepted_vectors: List[List[float]] = []

        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            texts = [item["text"] for item in chunk]
            embeddings = self.get_embeddings(texts=texts, model=model)

            batch_to_upsert = []
            for item, emb in zip(chunk, embeddings):
                meta = item.get("metadata", {}).copy() if item.get("metadata") else {}
                meta["text"] = item["text"]

                if filter_duplicates:
                    # 1. Проверка на дубликаты среди уже отобранных векторов в текущей сессии
                    is_in_batch_dup = False
                    for prev_item, prev_vec in zip(accepted_records, accepted_vectors):
                        sim = compute_cosine_similarity(emb, prev_vec)
                        if sim >= threshold:
                            is_in_batch_dup = True
                            skipped_duplicates.append({
                                "id": item["id"],
                                "text": item["text"],
                                "reason": "in_batch_duplicate",
                                "similarity": sim,
                                "matched_id": prev_item["id"],
                            })
                            logger.info(
                                f"Пропущен дубликат в батче: '{item['id']}' сходен с '{prev_item['id']}' "
                                f"({sim:.4f} >= {threshold})"
                            )
                            break

                    if is_in_batch_dup:
                        continue

                    # 2. Проверка на наличие похожего вектора в существующей базе данных Pinecone
                    db_dup = self.find_duplicate(
                        vector=emb,
                        namespace=namespace,
                        threshold=threshold,
                    )
                    if db_dup:
                        skipped_duplicates.append({
                            "id": item["id"],
                            "text": item["text"],
                            "reason": "database_duplicate",
                            "similarity": db_dup["score"],
                            "matched_id": db_dup["id"],
                            "matched_text": db_dup.get("text"),
                        })
                        logger.info(
                            f"Пропущен дубликат из базы: '{item['id']}' сходен с '{db_dup['id']}' "
                            f"({db_dup['score']:.4f} >= {threshold})"
                        )
                        continue

                accepted_records.append(item)
                accepted_vectors.append(emb)
                batch_to_upsert.append({
                    "id": item["id"],
                    "values": emb,
                    "metadata": meta,
                })

            if batch_to_upsert:
                res = self.upsert_vectors(vectors=batch_to_upsert, namespace=namespace)
                total_upserted += res.get("upserted_count", len(batch_to_upsert))
                total_batches += 1

        return {
            "upserted_count": total_upserted,
            "skipped_duplicates_count": len(skipped_duplicates),
            "duplicates": skipped_duplicates,
            "batches_count": total_batches,
            "threshold": threshold,
        }

    def upsert_text(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        filter_duplicates: bool = True,
        duplicate_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Высокоуровневый метод: создание эмбеддинга текста через OpenRouter и
        сохранение вектора и исходного текста в Pinecone с автоматической фильтрацией дубликатов.

        Args:
            id (str): Уникальный идентификатор записи.
            text (str): Текст документа/сообщения.
            metadata (Optional[Dict[str, Any]]): Дополнительные метаданные.
                Поле "text" автоматически добавляется/перезаписывается исходным текстом.
            namespace (Optional[str]): Пространство имен Pinecone.
            model (Optional[str]): Модель эмбеддингов OpenRouter.
            filter_duplicates (bool): Автоматически проверять наличие дубликатов перед записью.
                По умолчанию True.
            duplicate_threshold (Optional[float]): Порог косинусного сходства для отсечения дубликатов.
                Если не указан, используется self.duplicate_threshold (по умолчанию 0.88).

        Returns:
            Dict[str, Any]: Результат выполнения:
                - При обнаружении дубликата (при filter_duplicates=True):
                    {
                        "status": "duplicate_skipped",
                        "upserted_count": 0,
                        "is_duplicate": True,
                        "duplicate": Dict[str, Any],
                        "threshold": float,
                        "id": str,
                        "text": str
                    }
                - При успешном добавлении:
                    {
                        "status": "success",
                        "upserted_count": 1,
                        "is_duplicate": False,
                        "id": str,
                        "text": str
                    }
        """
        threshold = (
            duplicate_threshold
            if duplicate_threshold is not None
            else self.duplicate_threshold
        )
        vector = self.get_embedding(text=text, model=model)

        if filter_duplicates:
            duplicate = self.find_duplicate(
                vector=vector,
                namespace=namespace,
                threshold=threshold,
            )
            if duplicate:
                logger.info(
                    f"Пропущен дубликат текста (сходство {duplicate['score']:.4f} >= {threshold}): "
                    f"совпадает с ID '{duplicate['id']}'"
                )
                return {
                    "status": "duplicate_skipped",
                    "upserted_count": 0,
                    "is_duplicate": True,
                    "duplicate": duplicate,
                    "threshold": threshold,
                    "id": id,
                    "text": text,
                }

        meta = metadata.copy() if metadata else {}
        meta["text"] = text

        record = {
            "id": id,
            "values": vector,
            "metadata": meta,
        }
        res = self.upsert_vectors(vectors=[record], namespace=namespace)
        return {
            "status": "success",
            "upserted_count": res.get("upserted_count", 1),
            "is_duplicate": False,
            "id": id,
            "text": text,
        }

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        deduplicate: bool = False,
        duplicate_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Высокоуровневый метод: поиск наиболее релевантных векторов по текстовому запросу.
        Автоматически превращает query_text в вектор через OpenRouter и ищет совпадения в Pinecone.

        Args:
            query_text (str): Текстовый поисковый запрос (например: "Где заказать пиццу пепперони?").
            top_k (int): Количество наиболее похожих результатов. По умолчанию 5.
            filter (Optional[Dict[str, Any]]): Фильтр по метаданным.
            include_metadata (bool): Возвращать ли метаданные найденных записей. По умолчанию True.
            namespace (Optional[str]): Пространство имен для поиска.
            model (Optional[str]): Модель эмбеддингов для векторизации запроса.
            deduplicate (bool): Исключать ли близкие дубликаты из результатов (по умолчанию False).
            duplicate_threshold (Optional[float]): Порог косинусного сходства для исключения дубликатов.

        Returns:
            List[Dict[str, Any]]: Список найденных результатов с полями 'id', 'score' и 'metadata'.
        """
        query_vector = self.get_embedding(text=query_text, model=model)
        return self.search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
            include_values=False,
            namespace=namespace,
            deduplicate=deduplicate,
            duplicate_threshold=duplicate_threshold,
        )

    # --------------------------------------------------------------------------
    # Вспомогательные приватные методы
    # --------------------------------------------------------------------------

    def _ensure_index_connected(self) -> None:
        """
        Проверка, подключен ли индекс. Если нет — возбуждает RuntimeError.
        """
        if self.index is None:
            raise RuntimeError(
                "Индекс Pinecone не подключен. Сначала вызовите connect_index(index_name) "
                "или передайте index_name при создании экземпляра PineconeManager / PineconeVectorClient."
            )


# Алиас класса для совместимости и лаконичности
PineconeManager = PineconeVectorClient

__all__ = [
    "PineconeManager",
    "PineconeVectorClient",
    "compute_cosine_similarity",
    "DEFAULT_DUPLICATE_THRESHOLD",
]
