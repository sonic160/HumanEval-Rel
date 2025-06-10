from .agent import Agent
from ..models.generation_model import GenerationModel
from .rag import RAG
from .summer import Summarizer


class KnowledgeDatabaseAgent(Agent):
    def __init__(
        self,
        llm_model: GenerationModel,
        rag_agent: RAG,
        summarizer_llm: GenerationModel,
        name: str = "knowledge agent",
    ):
        super().__init__(name)
        self.model = llm_model
        self.rag = rag_agent
        self.summarizer = Summarizer(summarizer_llm)

    def generate_answer(self, question: str):

        # Documents selection (a concatenation of retrieved documents)
        infor = self.rag.corresponding_documents(question)

        # Preparing the helping_info using the query and the selected documents
        helping_info = self.summarizer.generate_answer(question, infor)
        prompt = KNOWLEDGE_PROMPT_HELP.format(
            question=question, helping_info=helping_info
        )

        output = self.model.generate_response(prompt)

        return helping_info + "\n" + output


class KnowledgeLLMAgent(Agent):
    def __init__(self, llm_model: GenerationModel, name: str = "knowledge agent"):
        super().__init__(name)
        self.model = llm_model

    def generate_answer(self, question: str):
        prompt = KNOWLEDGE_PROMPT.format(question=question)
        output = self.model.generate_response(prompt)
        return output


KNOWLEDGE_PROMPT_HELP = """
You are an expert in the domain of risk and reliability. 
Please answer the following question briefly:
{question}
given that :
{helping_info}
"""

KNOWLEDGE_PROMPT = """
You are an expert in the domain of risk and reliability. 
Please answer the following question briefly:
{question}
"""
