from .agent import Agent
from ..models.generation_model import GenerationModel

class Summerizer:
    def __init__(self, llm_model: GenerationModel):
        self.model = llm_model

    def generate_answer(self, question:str ,infor: str):
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



