from encoder.interface import Encoder

from clip_client import Client


class ClipAsServiceEncoder(Encoder):
    def __init__(self, config: dict):
        super(ClipAsServiceEncoder, self).__init__(config)
        credential = {}
        if "clip_access_token" in config:
            credential['Authorization'] = config["clip_access_token"]
        self.client = Client(
            config["clip_server"],
            credential
        )

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        print(texts)
        return self.client.encode(texts).tolist()
