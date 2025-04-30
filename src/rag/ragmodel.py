from core.model import HuggingFace
from .database import DataBase
from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph



class RAGModel(HuggingFace):
    """
    RAGModel class that inherits from the HuggingFace class.
    This class is used to initialize a RAG model with a specified database and model path.
    It builds the state graph for the RAG model and provides a method to generate responses.
    """
    def __repr__(self):
        return f"RAGModel({self.model_name}, {self.database.name})"
    def __init__(
        self,
        model_name: str,
        # rag settings
        database: DataBase,
        # settings specific to the HuggingFace model class
        model_path: str,
        quantization_config=None,
        cache_dir=None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.quantization_config = quantization_config
        self.cache_dir = cache_dir
        self.database = database
        super().__init__(model_name, model_path, quantization_config, cache_dir)
        self.model_name = model_name+"_" + self.database.name

    def model_init(self) -> None:
        print("[RAGModel] Initializing rag database")
        self.database.build()
        print("[RAGModel] Initializing HuggingFace model")
        super().model_init()
        
        parent_generate = super(RAGModel, self).generate
        print("[RAGModel] Building state graph")

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
            # huggingface model settings
            max_tokens: int = 1000
            top_p: float = 0.95
            top_k: int = 60
            temperature: float
            prompt_prefix: str 
            prompt_suffix: str
            stream: bool

        # Define application steps
        def retrieve(state: State):
            retrieved_docs = self.database.vector_store.similarity_search(
                state["question"],
                k=3,
            )
            return {"context": retrieved_docs}
        def invoke_llm(state: State):
            docs_content = ""
            for i, doc in enumerate(state["context"]):
                docs_content += f"Document {i}: {doc.page_content}\n\n"
            prompt = f'RAG documents given to you {docs_content} \n\n, question:\n {state["question"]}'
            response = parent_generate(  # use the generate method from the HuggingFace class
                    prompt,
                    state["max_tokens"],
                    state["top_p"],
                    state["top_k"],
                    state["temperature"],
                    state["prompt_prefix"],
                    state["prompt_suffix"],
                    state["stream"],
                )
            return {"answer": response}

        graph_builder = StateGraph(State).add_sequence([retrieve, invoke_llm])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        top_k: int = 60,
        temperature: float = 0.3,
        prompt_prefix: Annotated[str, "passed by the prompter"] = "",
        prompt_suffix: Annotated[str, "passed by the prompter"] = "",
        stream: bool = True,
    ):
        response = self.graph.invoke(
            {
                "question": prompt,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "prompt_prefix": prompt_prefix,
                "prompt_suffix": prompt_suffix,
                "stream": stream,
            }
        )
        return response["answer"]



