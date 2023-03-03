from indexer.interface import Indexer, VectorDistance
from typing import Union
import msgpack
import faiss
import os
import numpy as np

FAISS_INDEX_KEY = "/index/faiss/indexes"
FAISS_DATA_KEY = "/index/faiss/"

class FaissIndexer(Indexer):
    def __init__(self, config: dict, database):
        super(FaissIndexer, self).__init__(config, database)
        self.base_path = config["data_path"]
        self.indexes = {}
        self.index_files = {}
        self.kv = database
        if self.kv.has(FAISS_INDEX_KEY):
            self.index_files = msgpack.unpackb(self.kv.get(FAISS_INDEX_KEY))
            for name in self.index_files:
                self.indexes[name] = faiss.read_index(self.index_files[name])

    def create_collection(self, name: str, vector_size: int, vector_distance: VectorDistance):
        if name in self.indexes:
            raise Exception("FaissIndexer: Collection Already Exists")
        self.index_files[name] = os.path.join(self.base_path, name + ".db")
        if os.path.exists(self.index_files[name]):
            self.indexes[name] = faiss.read_index(self.index_files[name])
        else:
            metric = faiss.METRIC_L2
            if vector_distance == VectorDistance.DOT:
                metric = faiss.METRIC_INNER_PRODUCT
            elif vector_distance == VectorDistance.COSINE:
                metric = faiss.METRIC_INNER_PRODUCT
            self.indexes[name] = faiss.IndexHNSWFlat(vector_size, 32, metric)

    def drop_collection(self, name: str):
        if name not in self.indexes:
            raise Exception("FaissIndexer: Collection Does Not Exist")
        if os.path.exists(self.index_files[name]):
            os.remove(self.index_files[name])
        del self.indexes[name]
        del self.index_files[name]

    def insert(self, name: str, points: list[list[float]], payloads: list[dict]) -> Union[list[str], list[int]]:
        start = self.indexes[name].ntotal
        ids = range(start, len(points))
        self.indexes[name].add(np.asarray(points).astype('float32'))
        for (index, ids) in enumerate(ids):
            self.kv.put(FAISS_DATA_KEY + name + "/" + str(ids), msgpack.packb(payloads[index]))

    def get_payloads(self, name: str, _id: Union[list[str], list[int]]) -> list[dict]:
        return [msgpack.unpackb(self.kv.get(FAISS_DATA_KEY + name + "/" + str(ids))) for ids in _id]

    def search(self, name: str, point: list[float], size: int) -> tuple[list[dict], Union[list[str], list[int]]]:
        _, I = self.indexes[name].search(np.asarray([point]).astype('float32'), size)
        ids = I[0]
        payloads = [msgpack.unpackb(self.kv.get(FAISS_DATA_KEY + name + "/" + str(i))) for i in ids]
        return payloads, ids

    def delete(self, name: str, _ids: Union[list[str], list[int]]):
        self.indexes[name].remove_ids(np.asarray(_ids).astype('int32'))
        for i in _ids:
            self.kv.delete(FAISS_DATA_KEY + name + "/" + str(i))

    def delete_all(self, name: str):
        self.drop_collection(name)

    def flush(self):
        for name in self.indexes:
            faiss.write_index(self.indexes[name], self.index_files[name])
        self.kv.put(FAISS_INDEX_KEY, msgpack.packb(self.index_files, use_bin_type=True))
