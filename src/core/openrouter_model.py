from typing import Annotated, List, Tuple
from openai import OpenAI
import os
from .model import Model

class OpenRouter(Model):
    """
    This class is used to interact with models available via the OpenRouter API
    """
    def __init__(
            self,
            model_name: str,
            api_key: str = None,
            base_url: str = "https://openrouter.ai/api/v1"
        ):
        """
        Initializes the OpenRouter model with the given model name and API key.g

        Args:
            model_name (str): The name of the model on OpenRouter (for example, "openai/gpt-3.5-turbo").
            api_key (str, optional): The API key for OpenRouter. If None, tries to get it from the OPENROUTER_API_KEY env var.
            base_url (str, optional): The base URL for the OpenRouter API. Defaults to "https://openrouter.ai/api/v1".
        """
        self.model_name = model_name
        # Use the provided API key or try to get it from the env var
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenRouter API key provided. Set the OPENROUTER_API_KEY environment variable or pass the api_key parameter.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

    def __str__(self):
        return f"{self.model_name}"

    def model_init(self):
        """
        For OpenRouter, it's not necessary to initialize a model locally since it's an API.
        This method is kept for interface compatibility.
        """
        pass

    def generate(self, 
                prompt: str, 
                max_tokens: int = 1000, 
                top_p: float = 0.95, 
                top_k: int = 60, 
                temperature: float = 0.3,
                prompt_prefix: Annotated[str, "passed by the prompter"] = "",
                prompt_suffix: Annotated[str, "passed by the prompter"] = "",
                stream: bool = False) -> Tuple[str, str]:
        """
        Generates code from a prompt using the OpenRouter API.

        Args:
            prompt (str): The prompt from which to generate code.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.
            top_p (float, optional): The cumulative probability for nucleus sampling. Defaults to 0.95.
            top_k (int, optional): The number of highest probability tokens to consider. Defaults to 60.
            temperature (float, optional): The temperature to control randomness. Defaults to 0.3.
            prompt_prefix (str, optional): Prefix to add before the prompt. Defaults to "".
            prompt_suffix (str, optional): Suffix to add after the prompt. Defaults to "".
            stream (bool, optional): Whether the response should be streamed. Defaults to False.

        Returns:
            tuple[str, str]: A tuple containing the extracted code and the full response.
        """
        full_prompt = f"{prompt_prefix}{prompt}{prompt_suffix}"
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            whole_answer = response.choices[0].message.content
            extracted_code = self.extract(whole_answer)
            return extracted_code, whole_answer
        except Exception as e:
            error_message = f"Erreur API OpenRouter: {str(e)}"
            print(error_message)
            return "", error_message
    
                
    def generate_batch(
        self, prompts: List[str], max_tokens: int = 100, top_p: float = 0.95, 
        top_k: int = 60, temperature: float = 0.3
    ) -> List[str]:
        """
        Generates code from a list of prompts using the OpenRouter API.

        Args:
            prompts (list): A list of prompts from which to generate code.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
            top_p (float, optional): The cumulative probability for nucleus sampling. Defaults to 0.95.
            top_k (int, optional): The number of highest probability tokens to consider. Defaults to 60.
            temperature (float, optional): The temperature value to control randomness. Defaults to 0.3.

        Returns:
            list: A list of generated code snippets corresponding to each prompt.
        """
        results = []
        
        # OpenRouter doesn't have a batch API, so we process each prompt individually
        for prompt in prompts:
            extracted_code, _ = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature
            )
            results.append(extracted_code)
            
        return results

    def extract(self, text):
        """
        Extracts code from the generated text.
        
        Args:
            text (str): The text containing the code.
            
        Returns:
            str: The extracted code.
        """
        return Model.extract_code(text)



if __name__ == "__main__":
    # Test de l'API OpenRouter
    try:
        # Initialisation du modèle
        model = OpenRouter(
            model_name="qwen/qwen3-235b-a22b:free",
            api_key="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"
            # api_key="your_api_key_here"  # Ou utiliser la variable d'environnement OPENROUTER_API_KEY
        )
        
        print(f"Modèle initialisé: {model}")
        
        # Test simple de génération
        test_prompt = "Write a simple Python function that adds two numbers"
        print(f"\nPrompt de test: {test_prompt}")
        
        extracted_code, full_response = model.generate(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        print(f"\nCode extrait:\n{extracted_code}")
        print(f"\nRéponse complète:\n{full_response}")
        
    except ValueError as e:
        print(f"Erreur d'initialisation: {e}")
    except Exception as e:
        print(f"Erreur lors du test: {e}")
