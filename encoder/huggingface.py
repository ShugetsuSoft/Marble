from encoder.interface import Encoder

from transformers import AutoTokenizer, CLIPTextModel


class HuggingfaceEncoder(Encoder):
    def __init__(self, config: dict):
        super(HuggingfaceEncoder, self).__init__(config)
        clip_model = "openai/clip-vit-base-patch32"
        if "clip_model" in config:
            clip_model = config["clip_model"]
        self.model = CLIPTextModel.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.pooler_output.tolist()
