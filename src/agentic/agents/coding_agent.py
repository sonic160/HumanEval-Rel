from .agent import Agent
from ..models.generation_model import GenerationModel
from ..special_tokens import *

class CodingAgent(Agent):
    """
    Agent specialized in generating Python code based on function signatures and step-by-step instructions.

    This agent formats a prompt combining the function signature and instructions,
    sends it to a language model for code generation, and extracts the resulting
    Python function code from the response.

    Args:
        llm_model (GenerationModel): Language model used to generate code.
        name (str): Optional name for the agent (default is "coding agent").
    """
    def __init__(self, llm_model: GenerationModel, name: str="coding agent"):
        super().__init__(name)
        self.model = llm_model

    def generate_code(self, signature: str, instructions: str):
        """
        Generate a complete Python function based on a function signature and instructions.

        Args:
            signature (str): The function signature to be implemented.
            instructions (str): Step-by-step instructions describing the function's behavior.

        Returns:
            str: The Python code for the function extracted from the language model's output.
        """
        prompt = GENERATION_PROMPT.format(signature = signature, instructions = instructions)
        output = self.model.generate_response(prompt)
        return self._extract_code(output)
    
    def _extract_code(self, output:str):
        """
        Internal method. Extracts the Python code snippet enclosed in triple backticks from the model
        output.

        If the code block is annotated with 'python', it is excluded from the result.

        Args:
            output (str): The raw output text from the language model.

        Returns:
            code (str): Extracted Python code as a string.
        """
        end = output.rfind('```')
        start = output.rfind('```', 0, end) + 3
        if output[start:].startswith('python'):
            start += len('python')

        return output[start: end]

GENERATION_PROMPT=f"""
You are a coding assitant tasked with helping scientits write Python code quickly and accuratly.
You will receive a function signature as well as step by step instruction of what the function should do
and your goal is to write the complete function.
- The function must be self contained.
- The function signature will be given between the tags {SIGNATURE_BEGIN} and {SIGNATURE_END}.
- The instructions to follow will be given between {INSTRUCTIONS_BEGIN} and {INSTRUCTIONS_END}.
- The final function must be written between '```'
- Be brief.

# Example:
Input: 
Write a function with the following signature
{SIGNATURE_BEGIN}
def p_f_interval(reliability: callable, t: float, delta_t: float) -> float:
    '''
    You will be given a callable reliability, which is a reliability function a lifetime distribution T, representing the lifetime of an item.
    You will be given two floats t and delta_t.
    Your task is to develop a python script to calculate the probability that the item fails either before t or after t + delta_t.
    Examples:
    >>> p_f_interval(lambda x: np.exp(-x), 1, 2)
    0.6819076271964216
    '''
{SIGNATURE_END}
    
The function should follow the following instructions:
{INSTRUCTIONS_BEGIN}
1. **Given Input**: `reliability` (callable), `t` (float), and `delta_t` (float)
2. **Evaluate CDF at T and T+ΔT**: Calculate `prob_before = reliability(t)` and `prob_after = reliability(t +
delta_t)`
3. **Calculate P**: P is the difference between prob_after and prob_before: `P = prob_after - prob_before`
4. **Return Result**: Return the calculated probability `P` as a float value
{INSTRUCTIONS_END}

Output:
```
def p_f_interval(reliability: callable, t: float, delta_t: float) -> float:
    '''
    You will be given a callable reliability, which is a reliability function a lifetime distribution T, representing the lifetime of an item.
    You will be given two floats t and delta_t.
    Your task is to develop a python script to calculate the probability that the item fails either before t or after t + delta_t.
    Examples:
    >>> p_f_interval(lambda x: np.exp(-x), 1, 2)
    0.6819076271964216
    '''
    prob_before = reliability(t)
    prob_after = reliability(t + delta_t)

    P = prob_after - prob_before

    return P
```

# Your Input:
Write a function of the following signature
{SIGNATURE_BEGIN}
{{signature}}
{SIGNATURE_END}

The function should follow the following instructions:
{INSTRUCTIONS_BEGIN}
{{instructions}}
{INSTRUCTIONS_END}
"""