from .agent import Agent
from ..models.generation_model import GenerationModel

class Summarizer:
    """
    Summarizer component that uses a language model to generate a summary based on a question
    and supporting context.

    This class formats the input using a summarization prompt template and leverages a
    language model to generate a concise and relevant summary.

    Args:
        llm_model (GenerationModel): A language generation model used to perform summarization.
    """
    def __init__(self, llm_model: GenerationModel):
        self.model = llm_model

    def generate_answer(self, question:str ,infor: str):
        """
        Generates a summary to a question using provided contextual information.

        Args:
            question (str): The question to summarize with respect to.
            infor (str): The context or supporting information used to summarize.

        Returns:
            str: The generated answer.
        """
        prompt = SUMMER_PROMPT.format(question=question, context=infor)
        output = self.model.generate_response(prompt)
        return output
    
SUMMER_PROMPT = """

Your are an assistant for summarization tasks. Summarize the given pieces of retrieved context with respect to the question to help another agent answer the question.

# Question: 
{question}

# Context: 
{context}
"""



