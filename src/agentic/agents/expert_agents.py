from .agent import Agent
from ..models.generation_model import GenerationModel
from .rag import RAG
from .summer import Summerizer


class KnowledgeDatabaseAgent(Agent):
    def __init__(
        self,
        llm_model: GenerationModel,
        rag_agent: RAG,
        summerizer_llm: GenerationModel,
        name: str = "knowledge agent",
    ):
        super().__init__(name)
        self.model = llm_model
        self.rag = rag_agent
        self.summerizer = Summerizer(summerizer_llm)

    def generate_answer(self, question: str):
        help_info = ""
        # Documents selection (a concatenation of retrieved documents)
        infor = self.rag.corresponding_documents_emb(question)

        # Preparing the help_info using the query and the selected documents
        helping_info = self.summerizer.generate_answer(question, infor)
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
