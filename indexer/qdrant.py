from indexer.interface import Indexer, VectorDistance
from typing import Union

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantIndexer(Indexer):
    def __init__(self, config: dict, database=None):
        super(QdrantIndexer, self).__init__(config, database)
        self.client = QdrantClient(host=config["qdrant_host"], port=config["qdrant_port"])

    def create_collection(self, name: str, vector_size: int, vector_distance):
        distance = Distance.EUCLID
        if vector_distance == VectorDistance.DOT:
            distance = Distance.DOT
        elif vector_distance == VectorDistance.COSINE:
            distance = Distance.COSINE
        self.client.recreate_collection(name, vectors_config=VectorParams(size=vector_size, distance=distance))

    def drop_collection(self, name: str):
        self.client.delete_collection(name)

    def insert(self, name: str, points: list[list[float]], payloads: list[dict]) -> list[str]:
        _uuids = [uuid.uuid1().hex for i in range(len(points))]
        _points = [PointStruct(
            id=_uuids[i],
            vector=points[i],
            payload=payloads[i]
        ) for i in range(len(points))]

        self.client.upsert(
            collection_name=name,
            points=_points
        )

        return _uuids

    def get_payloads(self, name: str, _id: list[str]) -> list[dict]:
        points = self.client.retrieve(
            collection_name=name, ids=_id, with_payload=True, with_vectors=False
        )
        return [i.payload for i in points]

    def search(self, name: str, point: list[float], size: int) -> tuple[list[dict], Union[list[str], list[int]]]:
        results = self.client.search(
            collection_name=name,
            query_vector=point,
            query_filter=None,
            append_payload=True,
            limit=size
        )
        payloads = []
        ids = []
        for i in results:
            payloads.append(i.payload)
            ids.append(i.id)
        return payloads, ids

    def delete(self, name: str, _ids: list[str]):
        self.client.delete(collection_name=name, points_selector=_ids)

    def delete_all(self, name: str):
        col = self.client.get_collection(collection_name=name)
        self.client.delete_collection(collection_name=name)
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=col.config.params.vectors,
            hnsw_config=col.config.hnsw_config
        )

    def get_size(self, name: str) -> int:
        return self.client.get_collection(name).points_count