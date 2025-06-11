from core.model import Model
from .models.generation_model import GenerationModel
from .models.embedding_model import EmbeddingModel
from enum import Enum
from .special_tokens import *
from .agents.coding_agent import CodingAgent
from .agents.expert_agents import KnowledgeDatabaseAgent, KnowledgeLLMAgent
from .agents.rag import RAG
import sys
import os
from .database import load_documents


class Entent(Enum):
    """
    Enumeration of possible reasoning outcomes used by the AgenticModel.

    Attributes:
        CallCodingAgent: Indicates the model should delegate to the coding agent.
        CallKnowledgeAgent: Indicates the model should delegate to the knowledge agent.
        GiveUp: Indicates that the model finished reasoning without providing a query to any agent.
    """
    CallCodingAgent = 0
    CallKnowledgeAgent = 1
    GiveUp = 2


class AgenticModel(Model):
    """
    A multi-agent language model that orchestrates reasoning, coding, and knowledge retrieval tasks.

    The AgenticModel coordinates interactions between multiple sub-models:
    - A reasoning model to determine the next action.
    - A knowledge agent for answering domain-specific questions, optionally using RAG (Retrieval-Augmented Generation).
    - A coding agent for final code generation.

    Attributes:
        resoning_model (GenerationModel): The model responsible for deciding the next action.
        coding_agent (CodingAgent): The agent responsible for generating code.
        knowledge_agent (KnowledgeAgent): The agent responsible for answering knowledge queries.
        use_rag (bool): Whether to use a RAG-based knowledge agent.
        selection (str): The method used for document selection in RAG ("emb", etc.).
    """
    def __init__(
        self,
        model_name: str,
        resoning_model: GenerationModel,
        coding_medel: GenerationModel,
        knowledge_model: GenerationModel,
        summarizer_model: GenerationModel,
        selector_model: EmbeddingModel,
        use_rag: bool = False,
        rag_data_base_path=None,
        selection: str = "emb",
    ):
        """
        Initializes the AgenticModel with specified sub-models and configurations.

        Args:
            model_name (str): Name of the agentic model.
            resoning_model (GenerationModel): Model used to generate reasoning steps.
            coding_medel (GenerationModel): Model used by the coding agent.
            knowledge_model (GenerationModel): Model used by the knowledge agent.
            use_rag (bool): Whether to use a RAG-based approach for the knowledge agent, or to `knowledge_model` to answer the queries directly.
            summarizer_model (GenerationModel): Required only if `use_rag = True`. Model for summarizing RAG content. 
            selector_model (EmbeddingModel): Required only if `use_rag = True`. Embedding model for document retrieval in RAG. 
            rag_data_base_path (str, optional): Required only if `use_rag = True`. Path to documents for RAG. 
            selection (str): Required only if `use_rag = True`. Method for selecting documents. It can be:
                - "emb" (default): The RAG will use the embeding of the tiltle of documents to answer queries.
                - "freq": The RAG will use word frequencies in the content of documents to answer queries.

        """
        super().__init__(model_name)

        self.resoning_model = resoning_model
        self.coding_agent = CodingAgent(coding_medel)
        self.use_rag = use_rag
        self.selection = selection

        if self.use_rag:
            documents = load_documents(rag_data_base_path)
            rag = RAG(
                embedding_model=selector_model,
                documents=documents,
                selection=self.selection,
            )
            self.knowledge_agent = KnowledgeDatabaseAgent(
                knowledge_model, rag, summarizer_llm=summarizer_model
            )
        else:
            self.knowledge_agent = KnowledgeLLMAgent(knowledge_model)

    def __str__(self):
        return f"{self.model_name}"

    def model_init(self):
        pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        top_k: int = 60,
        temperature: float = 0.3,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        stream: bool = False,
    ) -> tuple[str, str]:
        """
        Generates a completion for the challenge (prompt).

        Args:
            prompt (str): The challenge to be solved.
            max_tokens (int): Added to stay complient with the inteface of `Model`. Has no effect.
            top_p (float): Added to stay complient with the inteface of `Model`. Has no effect.
            top_k (int): Added to stay complient with the inteface of `Model`. Has no effect.
            temperature (float): Added to stay complient with the inteface of `Model`. Has no effect.
            prompt_prefix (str): Added to stay complient with the inteface of `Model`. Has no effect.
            prompt_suffix (str): Added to stay complient with the inteface of `Model`. Has no effect.
            stream (bool): Added to stay complient with the inteface of `Model`. Has no effect.

        Returns:
            (final_output, reasoning_step): A tuple containing the final code output and the full
                             annotated reasoning prompt.
                    - final_output (str): the completion generated by the model.
                    - reasoning_step (str): the reasoning steps leading to the final answer.
        """

        final_output = ""
        challenge = prompt
        prompt = REASONING_PROMPT.format(challenge=prompt)
        while True:
            print("reasoning")
            entent, query = self._run_reasoning_step(prompt)

            if entent == Entent.CallKnowledgeAgent:
                print(f"querying the knowledge agent for: {query}")
                answer = self.knowledge_agent.generate_answer(query)
                print(f"Expert responds:\n {answer}")
                prompt += f"\n {KNOWLEDGE_RESULT_BEGIN} {answer} {KNOWLEDGE_RESULT_END}"

            elif entent == Entent.CallCodingAgent:
                print(f"querying the coding agent for: {query}")
                final_output = self.coding_agent.generate_code(challenge, query)
                print(f"Coding agent responds:\n {final_output}")
                break

            elif entent == Entent.GiveUp:
                print(
                    "Model finished generation without providing the instructions to the coding agent"
                )
                break

        return final_output, prompt

    def generate_batch(
        self, prompts, max_tokens=100, top_p=0.95, top_k=60, temperature=0.3
    ) -> list[str]:
        """
        Generates outputs for a batch of prompts. 

        Args:
            prompts (list[str]): List of prompts to process.
            max_tokens (int): Added to stay complient with the inteface of `Model`. Has no effect.
            top_p (float): Added to stay complient with the inteface of `Model`. Has no effect.
            top_k (int): Added to stay complient with the inteface of `Model`. Has no effect.
            temperature (float): Added to stay complient with the inteface of `Model`. Has no effect.

        Returns:
            list[str]: List of generated outputs.
        """
        return [self.generate(prompt)[0] for prompt in prompts]

    def _run_reasoning_step(self, prompt) -> tuple[Entent, str]:
        """
        This is an internal method. Executes a single reasoning step using the reasoning model to determine the next action.

        Args:
            prompt (str): The current prompt including reasoning context.

        Returns:
            tuple[Entent, str]: A tuple containing:
                - The intent (`Entent`) indicating the next action.
                - The extracted query or instruction for the corresponding agent.
        """
        st_tokens = [CODING_QUERY_END, KNOWLEDGE_QUERY_END]

        output = self.resoning_model.generate_response(
            prompt, stop_tokens=st_tokens, max_tokens=1000
        )

        if CODING_QUERY_BEGIN in output:
            start_query = output.find(CODING_QUERY_BEGIN) + len(CODING_QUERY_BEGIN)
            end_query = output.find(CODING_QUERY_END, start_query)

            if end_query == -1:
                end_query = len(output)
                output += CODING_QUERY_END

            entent = Entent.CallCodingAgent
            instructions = output[start_query:end_query]
            return entent, instructions
        elif KNOWLEDGE_QUERY_BEGIN in output:
            entent = Entent.CallKnowledgeAgent
            start_query = output.find(KNOWLEDGE_QUERY_BEGIN) + len(
                KNOWLEDGE_QUERY_BEGIN
            )
            end_query = output.find(KNOWLEDGE_QUERY_END, start_query)

            if end_query == -1:
                end_query = len(output)
                output += KNOWLEDGE_QUERY_END

            query = output[start_query:end_query]
            return entent, query

        else:
            return Entent.GiveUp, ""


REASONING_PROMPT = (
    "You are a reasoning assistant with the ability to break down complex challenges into clear, step-by-step instructions(reasoning steps). "
    f"You solve the user's challenge (query) accurately by providing instructions, which will then be sent to another agent that translates them into code. Your final answer should be a set of instructions formatted as: {CODING_QUERY_BEGIN} ... your instructions here ... {CODING_QUERY_END}. You also have special tools:\n\n"
    "Make sure your each code query is self-contained and does not require any external information.\n\n"
    f"You have access to an expert agent capable of answering any questions you may have. Simply use the following format: {KNOWLEDGE_QUERY_BEGIN} ...your query... {KNOWLEDGE_QUERY_END}\n"
    f"The system will then analyze your previous reasoning and answer your query in the following format: {KNOWLEDGE_RESULT_BEGIN} ...answer results... {KNOWLEDGE_RESULT_END}\n\n"
    "You can repeat calling the tools multiple times if necessary.\n\n"
    "Once you have all the information you need, continue your reasoning.\n\n"
    "Example1:\n"
    'Challenge: "def p_f_interval(reliability: callable, t: float, delta_t: float) -> float:\n\n"'
    """ You will be given a callable reliability, which is a reliability function a lifetime distribution T, representing the lifetime of an item. You will be given two floats t and delta_t. Your task is to develop a python script to calculate the probability that the item fails either before t or after t + delta_t.
            Examples:
            >>> p_f_interval(lambda x: np.exp(-x), 1, 2)
            0.6819076271964216
        """
    "\n\n"
    "To solve the challenge, I need to better define a reliability function and to know how to link it to failure probabilities\n\n"
    f"{KNOWLEDGE_QUERY_BEGIN}What is a reliability function, and how is the probability of failure related to the reliability function? {KNOWLEDGE_QUERY_END}\n\n"
    f"{KNOWLEDGE_RESULT_BEGIN}\n"
    "The reliability function R(t) gives the probability that an item survives beyond time t: R(t) = P(T > t)\n"
    "Therefore, the probability of failure before time t is: P(T ≤ t) = 1 - R(t)\n"
    "Similarly, the probability of survival after time (t + delta_t) is: P(T > t + delta_t) = R(t + delta_t)\n"
    "Thus, the probability that the item fails either before t or after t + delta_t is: P(T ≤ t) + P(T > t + delta_t) = (1 - R(t)) + R(t + delta_t)\n"
    "Note: We assume the two events (fail before t, fail after t + delta_t) are disjoint.\n"
    f"{KNOWLEDGE_RESULT_END}\n\n"
    "Now, I fully understand the math.\n\n"
    "Next, I will write the instructions for the solution:\n"
    # "Next, I will write the pseudo-code for the solution:\n"
    # "Function p_f_interval(reliability, t, delta_t):\n"
    # "   fail_before_t = 1 - reliability(t)\n"
    # "   survive_after_t_plus_delta = reliability(t + delta_t)\n"
    # "   result = fail_before_t + survive_after_t_plus_delta\n"
    # "   return result\n\n"
    "Assistant:\n"
    f"{CODING_QUERY_BEGIN}\n Write a Python function p_f_interval(reliability: callable, t: float, delta_t: float) -> float that:\n"
    "Calculates 1 - reliability(t)\n"
    "Calculates reliability(t + delta_t)\n"
    f"Returns the sum of these two quantities. \n{CODING_QUERY_END}\n\n"
    "Remember:\n"
    f"- Use {KNOWLEDGE_QUERY_BEGIN} to ask a question to the expert agent, and end with {KNOWLEDGE_QUERY_END}.\n"
    f"- Use {CODING_QUERY_BEGIN} to deliver your final answer, which should be a set of instructions, and end with {CODING_QUERY_END}.\n"
    "- When done asking, continue your reasoning.\n\n"
    f"here is your challenge now :{{challenge}} , follow the above instruction to solve it.\n\n"
)
