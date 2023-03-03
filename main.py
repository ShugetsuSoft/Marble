from fastapi import FastAPI, Request, Depends

from encoder import ClipAsServiceEncoder
from indexer import QdrantIndexer, VectorDistance
from database import RedisDatabase
import msgpack
from pydantic import BaseModel
import json


class CreateRequest(BaseModel):
    should_index: list[str]


with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.loads(f.read())

MARBLE_INDEX_SHOULD_INDEX = "/mb/should/"


class Marble():
    def __init__(self, config: dict):
        self.config = config
        self.database = RedisDatabase(config)
        self.encoder = ClipAsServiceEncoder(config)
        self.indexer = QdrantIndexer(config, self.database)

    def create_index(self, name, should_index):
        self.database.put(MARBLE_INDEX_SHOULD_INDEX + name, msgpack.packb(should_index, use_bin_type=True))
        self.indexer.create_collection(name, self.config["vector_size"], VectorDistance.COSINE)

    def insert(self, name, documents: list[dict]) -> list[str]:
        should_index = self.get_indexed_fields(name)
        if not should_index:
            return None
        texts = []
        for d in documents:
            text = []
            for k in d:
                if k in should_index and type(d[k]) is str:
                    text.append(d[k])
            texts.append("\n".join(text))

        embeddings = self.encoder.encode_text(texts)
        return self.indexer.insert(name, embeddings, documents)

    def get_indexed_fields(self, name):
        data = self.database.get(MARBLE_INDEX_SHOULD_INDEX + name)
        if not data:
            return None
        should_index = set(msgpack.unpackb(data))
        return should_index

    def search(self, name, text: str, k: int):
        embedding = self.encoder.encode_text([text])[0]
        docs, ids = self.indexer.search(name, embedding, k)
        for (index, ida) in enumerate(ids):
            docs[index]["_id"] = ida
        return docs

    def delete(self, name, ids):
        self.indexer.delete(name, ids)

    def drop_index(self, name):
        self.database.delete(MARBLE_INDEX_SHOULD_INDEX + name)
        self.indexer.drop_collection(name).points_count

    def get_index_size(self, name):
        return self.indexer.get_size(name)


marble = Marble(CONFIG)
app = FastAPI()


@app.get("/")
def main():
    return "hello world"


@app.get("/{index_name}/stat")
def get_index(index_name: str):
    indexed_fields = marble.get_indexed_fields(index_name)
    if not indexed_fields:
        return {
            "err": "not exists"
        }
    return {
        "name": index_name,
        "size": marble.get_index_size(index_name),
        "indexed_fields": indexed_fields
    }


async def get_body_json(request: Request):
    return await request.json()


@app.post("/{index_name}")
def insert(index_name: str, documents: list[dict] = Depends(get_body_json)):
    marble.insert(index_name, documents)
    return {"err": None}


@app.get("/{index_name}")
def query(index_name: str, text: str, limit: int = 10):
    return marble.search(index_name, text, limit)


@app.put("/{index_name}")
def create(index_name: str, data: CreateRequest):
    marble.create_index(index_name, data.should_index)
    return {"err": None}


@app.delete("/{index_name}")
def delete(index_name: str):
    marble.drop_index(index_name)
    return {"err": None}
