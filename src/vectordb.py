from abc import ABC, abstractmethod
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    PointStruct,
    UpdateResult,
    VectorParams,
)

from src.utils import BOLD, END, log


# Define an abstract class VectorDB
class VectorDB(ABC):
    def __init__(self, collections: list[dict[str, str | int]]):
        self.collections = collections
        pass

    @abstractmethod
    def add(self, collection_name: str, entity_id: str, vector: list[float], sequence: str | None = None) -> None:
        pass

    @abstractmethod
    def get(
        self, collection_name: str, search_input: str | None = None, search_field: str = "id", limit: int = 5
    ) -> list[Any]:
        pass

    @abstractmethod
    def search(self, collection_name: str, vector: str) -> list[tuple[str, float]]:
        pass


# https://qdrant.tech/documentation/quick-start
class QdrantDB(VectorDB):
    def __init__(
        self,
        collections: list[dict[str, str | int]],
        recreate: bool = False,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
    ):
        super().__init__(collections)
        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        if len(collections) < 1:
            raise ValueError('Provide at least 1 collection, e.g. [{"name": "my_collec", "size": 512}]')

        # TODO: add indexing for id and sequence in payload
        if recreate:
            for collection in collections:
                self.client.recreate_collection(
                    collection_name=collection["name"],
                    vectors_config=VectorParams(size=collection["size"], distance=Distance.DOT),
                )
                self.client.create_payload_index(collection["name"], "id", "keyword")
        else:
            try:
                log.info(
                    f"💊 {self.client.get_collection('drug').points_count} vectors in the {BOLD}drug{END} collection"
                )
                log.info(
                    f"🎯 {self.client.get_collection('target').points_count} vectors in the {BOLD}target{END} collection"
                )
            except Exception as e:
                log.info(f"⚠️ Collection not found: {e}, recreating the collections")
                for collection in collections:
                    self.client.recreate_collection(
                        collection_name=collection["name"],
                        vectors_config=VectorParams(size=collection["size"], distance=Distance.DOT),
                        # Qdrant supports Dot, Cosine and Euclid
                    )
                    self.client.create_payload_index(
                        collection["name"],
                        "id",
                        {
                            "type": "text",
                            "tokenizer": "word",
                            "min_token_len": 2,
                            "max_token_len": 30,
                            # "lowercase": True
                        },
                    )

    def add(self, collection_name: str, item_list: list[str]) -> UpdateResult:
        points_count = self.client.get_collection(collection_name).points_count
        points_list = [
            PointStruct(id=points_count + i + 1, vector=item["vector"], payload=item["payload"])
            for i, item in enumerate(item_list)
        ]
        # PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points_list,
        )
        return operation_info

    # Get the embeddings for a specific entity ID
    def get(
        self, collection_name: str, search_input: str | None = None, search_field: str = "id", limit: int = 5
    ) -> list[Any]:
        # if search_input and ":" in search_input:
        #     search_input = search_input.split(":", 1)[1]
        search_result = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(should=[FieldCondition(key=search_field, match=MatchText(text=search_input))])
            if search_input
            else None,
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        return search_result[0]

    def search(
        self, collection_name: str, vector: str, search_input: str | None = None, limit: int = 10
    ) -> list[Any] | None:
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=Filter(must=[FieldCondition(key="id", match=MatchText(value=search_input))])
            if search_input
            else None,
            limit=limit,
        )
        return search_result[0]


def init_vectordb(collections: list[dict[str, str]], recreate: bool = False, api_key: str = "TOCHANGE"):
    qdrant_url = "qdrant.137.120.31.148.nip.io"
    return QdrantDB(collections=collections, recreate=recreate, host=qdrant_url, port=443, api_key=api_key)
