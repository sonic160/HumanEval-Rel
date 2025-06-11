from .agent import Agent
from ..models.generation_model import GenerationModel
from .rag import RAG
from .summer import Summarizer


class KnowledgeDatabaseAgent(Agent):
    """
    Knowledge agent that leverages a Retrieval-Augmented Generation (RAG) pipeline and a summarizer
    to answer questions using retrieved contextual information.

    This agent first retrieves relevant documents using a RAG system, then summarizes those documents 
    in relation to the question, and finally uses an LLM to generate the answer based on the summarized context.

    Args:
        llm_model (GenerationModel): The language model used to generate the final answer.
        rag_agent (RAG): The retrieval component used to fetch relevant documents based on the question.
        summarizer_llm (GenerationModel): The language model used by the summarizer.
        name (str, optional): Name of the agent. Defaults to "knowledge agent".
    """
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
        """
        Generates an answer to a given question using retrieved and summarized context.

        The method performs the following:
        1. Retrieves documents using the RAG agent.
        2. Summarizes them with respect to the question.
        3. Uses an LLM to generate a final answer using the summarized information.
        4. If the answer is short, the summary is prepended to provide more context.

        Args:
            question (str): The input question to answer.

        Returns:
            str: The generated answer, possibly prefixed by the summarized context if the output is short.
        """
        # Documents selection (a concatenation of retrieved documents)
        infor = self.rag.corresponding_documents(question)

        # Preparing the helping_info using the query and the selected documents
        helping_info = self.summarizer.generate_answer(question, infor)
        prompt = KNOWLEDGE_PROMPT_HELP.format(
            question=question, helping_info=helping_info
        )

        output = self.model.generate_response(prompt)
        if len(output.split(" ")) < 30 :
            return helping_info + "\n" + output
        return output


class KnowledgeLLMAgent(Agent):
    """
    Knowledge agent that directly uses an LLM to answer questions without using external knowledge retrieval.

    Args:
        llm_model (GenerationModel): The language model used to generate answers.
        name (str, optional): Name of the agent. Defaults to "knowledge agent".
    """
    def __init__(self, llm_model: GenerationModel, name: str = "knowledge agent"):
        super().__init__(name)
        self.model = llm_model

    def generate_answer(self, question: str):
        """
        Generates an answer to a given question using the language model.

        Args:
            question (str): The input question to answer.

        Returns:
            str: The generated answer.
        """
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
