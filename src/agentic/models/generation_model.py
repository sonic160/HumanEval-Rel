from abc import ABC, abstractmethod
import os

import torch
from litellm import completion
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


class GenerationModel(ABC):
    """
    Abstract base class for language generation models.

    Args:
        model_name (str): Identifier or name of the language model.
    """
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        temperature: int = 0.3,
        stop_tokens: list = None,
    ) -> str:
        """
        Generates a response from the model given an input prompt.

        Args:
            prompt (str): The input prompt for generation.
            max_tokens (int): Maximum number of tokens to generate.
            top_p (float): Top-p (nucleus) sampling parameter.
            temperature (float): Sampling temperature.
            stop_tokens (list, optional): List of stop tokens to truncate the response.

        Returns:
            str: The generated text response.
        """


class LitellmModel(GenerationModel):
    """
    A concrete implementation of GenerationModel using the LiteLLM API.

    A list of the supported models can be found [here](https://github.com/BerriAI/litellm).

    Args:
        model_name (str): Name or identifier of the model (e.g., 'gpt-3.5-turbo').
        api_token (str, optional): API key for authenticating with LiteLLM.
    """
    def __init__(self, model_name="gpt-3.5-turbo", api_token=None):
        self.model_name = model_name
        self.api_token = api_token

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        temperature: int = 0.3,
        stop_tokens: list = None,
    ) -> str:
        """
        Generates a response from the LiteLLM model given an input prompt.

        Args:
            prompt (str): The input prompt for generation.
            max_tokens (int): Maximum number of tokens to generate.
            top_p (float): Cumulative probability for nucleus sampling.
            temperature (float): Sampling temperature.
            stop_tokens (list, optional): List of tokens to stop generation.

        Returns:
            str: The generated text response.
        """
        # print number of token with autotokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # print("===========token count", len(tokenizer.tokenize(messages[0]["content"])))
        # print(f"Prompt : \n {messages[0]['content']}")

        messages = [{"role": "user", "content": prompt}]

        response = completion(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            api_key=self.api_token,
        )

        response = response.choices[0].message.content

        return response


class HuggingFace(GenerationModel):
    # This class can be used to use an open source model that is available on the HuggingFace library
    def __init__(
        self,
        model_name: str,
        model_path: str = None,
        quantization_config=None,
        cache_dir=None,
    ):
        """
        Initializes the HuggingFace model with the given model name and path.

        Args:
            model_name (str): The name of the model.
            model_path (str): The full path of the model on HuggingFace, e.g. croissantllm/CroissantLLMBase.
            quantization_config (optional): A QuantizationConfig object to specify the quantization configuration if the model is too big for your computer. Defaults to None.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.quantization_config = quantization_config
        self._cache_dir = cache_dir

        self.model_init()

    def model_init(self):
        """
        Initializes the model by loading the tokenizer and the model itself.

        Parameters:
        - cachedir (str): The directory path to cache the pretrained model and tokenizer. Defaults to None.

        Returns:
        - None
        """
        # We initialize the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            cache_dir=self._cache_dir, 
            use_fast=not (self.model_name in ("facebookMobileLLM-125M","facebookMobileLLM-1B")),# has pb with this particular model
            trust_remote_code=True
        )

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # We check if a quantization config was passed.
        if device == torch.device('cuda'):
            if self.quantization_config is None:
                self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                cache_dir=self._cache_dir,
                device_map='auto',  
                attn_implementation='flash_attention_2', 
                torch_dtype=torch.float16,
                trust_remote_code=True
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                cache_dir=self._cache_dir,
                device_map='auto',  
                attn_implementation='flash_attention_2', 
                torch_dtype=torch.float16,
                quantization_config=self.quantization_config
                ).to(device)
        elif device == torch.device('cpu'):
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                cache_dir=self._cache_dir,
                torch_dtype=torch.float16
                ).to(device)
        else:
            raise Exception(f"Unexpected device: {device}")
        #We the add padding tokens if none is defined
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self._model.resize_token_embeddings(len(self._tokenizer))

    def generate_response(
        self,
        prompt,
        max_tokens=1000,
        top_p=0.95,
        temperature=0.3,
        stop_tokens=None,
    ):
        """
        Generates a response for a given prompt using the Hugging Face model.

        Args:
            prompt (str): The prompt to generate code from.
            max_tokens (int, optional): Maximum number of tokens to generate.
            top_p (float, optional): Cumulative probability for nucleus sampling.
            top_k (int, optional): Number of top tokens to sample from.
            temperature (float, optional): Simpling tempreture.
            stop_tokens (list[str], optional): If provided, generation will stop at the first stop token.

        Returns:
            str: The generated response.
        """
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self._model.device)

        print("Generating LLM answer.")
        tokens = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self._tokenizer.eos_token_id,
            stop_strings=stop_tokens,
            tokenizer=self._tokenizer,
        )
        output = self._tokenizer.decode(tokens[0], skip_special_tokens=True)

        print("LLM answer generated.")

        if output.startswith(prompt):
            output = output[len(prompt) :].strip()

        print(" Here is the answer")
        print(" ")
        print(output)
        print(" ")
        print(" ")
        print("Answer finished ")

        return output
