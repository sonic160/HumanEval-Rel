import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    def __init__(self, model_name="Alibaba-NLP/gte-base-en-v1.5", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )

    def generate_embedding(self, input_text: str) -> list:
        """Generate embedding for a single text"""
        return self.generate_embeddings([input_text])[0]

    def generate_embeddings(self, input_texts: list) -> list:
        """Generate embeddings for a list of texts"""
        batch_dict = self.tokenizer(
            input_texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0].cpu().tolist()

        return embeddings
