import os
import logging
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка базового логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PineconeVectorClient:
    """
    Клиент для взаимодействия с векторной базой данных Pinecone
    с поддержкой генерации эмбеддингов через OpenRouter.

    Предоставляет методы для:
      - подключения к базе и индексам Pinecone,
      - записи (upsert) векторов и текстов,
      - чтения (fetch) векторов по ID,
      - семантического поиска (query / search) векторов и текстовых запросов,
      - удаления (delete) векторов по ID, фильтрам или полной очистки namespace.
    """

    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        default_embedding_model: str = "openai/text-embedding-3-small",
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

    def list_indexes(self) -> List[str]:
        """
        Получение списка названий всех доступных индексов в аккаунте Pinecone.

        Returns:
            List[str]: Список имен индексов (например, ['rag-demo-index', 'test2']).

        Raises:
            Exception: При ошибке обращения к API Pinecone.
        """
        try:
            indexes = self.pc.list_indexes()
            return [idx.name for idx in indexes]
        except Exception as e:
            logger.error(f"Ошибка при получении списка индексов: {e}")
            raise

    def connect_index(self, index_name: str) -> Any:
        """
        Подключение к существующему индексу Pinecone по его названию.

        Args:
            index_name (str): Название индекса Pinecone.

        Returns:
            Any: Экземпляр подключенного индекса Pinecone (Index).

        Raises:
            ValueError: Если индекс с таким именем отсутствует в аккаунте.
            Exception: При ошибке подключения к индексу.
        """
        try:
            available_indexes = self.list_indexes()
            if index_name not in available_indexes:
                raise ValueError(
                    f"Индекс '{index_name}' не найден. Доступные индексы: {available_indexes}"
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
            kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": include_metadata,
                "include_values": include_values,
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
                if include_values:
                    item["values"] = getattr(match, "values", None)
                results.append(item)

            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске векторов: {e}")
            raise

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
    ) -> Dict[str, Any]:
        """
        Пакетная запись текстов: вычисляет эмбеддинги для списка записей и
        сохраняет их в Pinecone с автоматическим батчингом.

        Args:
            items (List[Dict[str, Any]]): Список словарей с текстами.
                Каждый элемент должен содержать:
                  - "id" (str): Уникальный ID.
                  - "text" (str): Текст документа/фразы.
                  - "metadata" (Optional[Dict[str, Any]]): Дополнительные метаданные (опционально).
            namespace (Optional[str]): Пространство имен Pinecone.
            model (Optional[str]): Модель эмбеддингов OpenRouter.
            batch_size (int): Размер пакета для генерации эмбеддингов и записи. По умолчанию 50.

        Returns:
            Dict[str, Any]: Словарь с результатами:
                - 'upserted_count' (int): Общее число сохраненных векторов.
                - 'batches_count' (int): Число обработанных пачек.

        Raises:
            ValueError: Если items пуст или структура элементов некорректна.
        """
        if not items:
            raise ValueError("Список items не может быть пустым.")

        total_upserted = 0
        total_batches = 0

        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            texts = [item["text"] for item in chunk]
            embeddings = self.get_embeddings(texts=texts, model=model)

            vectors_to_upsert = []
            for item, emb in zip(chunk, embeddings):
                meta = item.get("metadata", {}).copy() if item.get("metadata") else {}
                meta["text"] = item["text"]
                vectors_to_upsert.append({
                    "id": item["id"],
                    "values": emb,
                    "metadata": meta,
                })

            res = self.upsert_vectors(vectors=vectors_to_upsert, namespace=namespace)
            total_upserted += res.get("upserted_count", len(vectors_to_upsert))
            total_batches += 1

        return {
            "upserted_count": total_upserted,
            "batches_count": total_batches,
        }

    def upsert_text(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Высокоуровневый метод: создание эмбеддинга текста через OpenRouter и
        сохранение вектора и исходного текста в Pinecone.

        Args:
            id (str): Уникальный идентификатор записи.
            text (str): Текст документа/сообщения.
            metadata (Optional[Dict[str, Any]]): Дополнительные метаданные.
                Поле "text" автоматически добавляется/перезаписывается исходным текстом.
            namespace (Optional[str]): Пространство имен Pinecone.
            model (Optional[str]): Модель эмбеддингов OpenRouter.

        Returns:
            Dict[str, Any]: Результат выполнения upsert_vectors.
        """
        vector = self.get_embedding(text=text, model=model)
        meta = metadata.copy() if metadata else {}
        meta["text"] = text

        record = {
            "id": id,
            "values": vector,
            "metadata": meta,
        }
        return self.upsert_vectors(vectors=[record], namespace=namespace)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
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
                "или передайте index_name при создании экземпляра PineconeVectorClient."
            )
