from enum import Enum, unique
from typing import Union

from database import interface

@unique
class VectorDistance(Enum):
    L2 = 0
    DOT = 1
    COSINE = 2


class Indexer(object):
    def __init__(self, config: dict, database: interface.Database):
        pass

    def create_collection(self, name: str, vector_size: int, vector_distance: VectorDistance):
        pass

    def drop_collection(self, name: str):
        pass

    def insert(self, name: str, points: list[list[float]], payloads: list[dict]) -> Union[list[str], list[int]]:
        pass

    def get_payloads(self, name: str, _id: Union[list[str], list[int]]) -> list[dict]:
        pass

    def search(self, name: str, point: list[float], size: int) -> tuple[list[dict], Union[list[str], list[int]]]:
        pass

    def delete(self, name: str, _ids: Union[list[str], list[int]]):
        pass

    def delete_all(self, name: str):
        pass

    def get_size(self, name: str) -> int:
        pass