import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    """
    Wrapper for a transformer-based sentence embedding model.

    Uses a pretrained Hugging Face model to generate embeddings for text inputs.

    Args:
        model_name (str): The name or path of the pretrained model.
        device (str): The device to load the model on (e.g., 'cpu' or 'cuda').
    """
    def __init__(self, model_name="Alibaba-NLP/gte-base-en-v1.5", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )

    def generate_embedding(self, input_text: str) -> list:
        """
        Generate an embedding for a single input text.

        Args:
            input_text (str): The input text to encode.

        Returns:
            list: The embedding vector for the input text.
        """
        return self.generate_embeddings([input_text])[0]

    def generate_embeddings(self, input_texts: list) -> list:
        """
        Generate embeddings for a list of input texts.

        Args:
            input_texts (list): A list of strings to encode.

        Returns:
            list: A list of embedding vectors corresponding to the input texts.
        """
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
