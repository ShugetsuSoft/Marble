from database.interface import Database
import redis


class RedisDatabase(Database):
    def __init__(self, config: dict):
        super(RedisDatabase, self).__init__(config)
        self.cli = redis.Redis.from_url(config["redis_uri"])

    def put(self, key: str, data: bytes):
        self.cli.set(key, data)

    def get(self, key: str) -> bytes:
        return self.cli.get(key)

    def delete(self, key: str):
        self.cli.delete(key)

