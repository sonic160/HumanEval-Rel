from ..core.model import Model
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

# Add project_root to the path so dataset_client becomes visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
class Entent(Enum):
    CallCodingAgent = 0
    CallKnowledgeAgent = 1
    GiveUp = 2


class AgenticModel(Model):
    def __init__(
        self,
        model_name: str,
        resoning_model: GenerationModel,
        coding_medel: GenerationModel,
        knowledge_model: GenerationModel,
        summarizer_model: GenerationModel,
        selector_model: EmbeddingModel,
        use_rag: bool = False,
        rag_data_base_path = None,
        selection: str = "emb",
    ):
        super().__init__(model_name)

        self.resoning_model = resoning_model
        self.coding_agent = CodingAgent(coding_medel)
        self.use_rag = use_rag
        self.selection = selection
        documents = load_documents(rag_data_base_path)

        if self.use_rag:
            rag = RAG(
                embedding_model=selector_model,
                documents=documents,
                selection=self.selection,
            )
            self.knowledge_agent = KnowledgeDatabaseAgent(
                knowledge_model, rag, summerizer_llm=summarizer_model
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
    ):
        return [self.generate(prompt)[0] for prompt in prompts]

    def _run_reasoning_step(self, prompt) -> tuple[Entent, str]:

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
