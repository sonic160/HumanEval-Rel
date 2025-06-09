from abc import ABC, abstractmethod
from typing import List, Tuple, Any




class System(ABC):
    """
    Abstract base class representing a system that can generate text,
    such as a language model. This class defines the interface
    that Prompter interacts with.
    """
    name : str 

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        stream: bool = False,
    ) -> Tuple[str, Any]:
        """
        Generates a completion for a single prompt.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): The maximum number of tokens to generate.
            top_p (float): The nucleus sampling probability.
            top_k (int): The top-k sampling parameter.
            temperature (float): The sampling temperature.
            prompt_prefix (str, optional): A prefix to add to the prompt. Defaults to "".
            prompt_suffix (str, optional): A suffix to add to the prompt. Defaults to "".
            stream (bool, optional): Whether to stream the output. Defaults to False.

        Returns:
            Tuple[str, Any]: A tuple containing the generated completion (str)
                             and the whole response object (Any) from the model.
        """
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generates completions for a batch of prompts.

        Args:
            prompts (List[str]): A list of input prompts.

        Returns:
            List[str]: A list of generated completions, corresponding to the input prompts.
        """
        pass