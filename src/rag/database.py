import os
from tqdm.auto import tqdm
from typing import Iterator, List
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .document_loaders import RagDocumentLoader, Separators


class DataBase(ABC):
    """
    Abstract base class for RAG (Retrieval-Augmented Generation) databases.
    
    Provides a framework for building, storing, and querying vector representations
    of documents for retrieval tasks. Documents are processed through loaders,
    chunked, and converted to vector embeddings.
    
    Subclasses must implement:
    - build_vectors: Create vector representations
    - build_vector_store: Create searchable store from vectors
    - query: Retrieve relevant documents for a query
    - save_vectors: Persist vectors to storage
    - load_vectors_from_file: Restore vectors from storage
    """

    def __init__(self, name: str, document_loaders: Iterator[RagDocumentLoader], force_rebuild : bool=False) -> None:
        self.name = name
        self.document_loaders = document_loaders
        self.force_rebuild = force_rebuild

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __str__(self) -> str:
        return self.__repr__()

    def chunk_documents(self) -> None:
        me = self.__class__.__name__
        print(f"[{me}] loading documents")
        documents: list[tuple[Document, Separators]] = []

        for loader in self.document_loaders:
            documents.extend(map(lambda x: (x, loader.separators), 
                                 loader.load()
                                 )
                                )

        print(f"[{me}] Loaded {len(documents)} documents")

        # Chunk the documents
        print(f"[{me}] Chunking documents")
        chunk_size = SentenceTransformer(self.embedding_model_name).get_max_seq_length()

        documents_chunks = []
        tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name, 
            model_max_length=512, 
            truncation=True
        )
        # Enforce a safer chunk size
        chunk_size = min(500, chunk_size)

        for doc, separators in documents:
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=separators,
            )
            documents_chunks.extend(text_splitter.split_documents([doc]))

        print(f"[{me}] Created {len(documents_chunks)} document chunks")
        self.chunks = documents_chunks

    def build(self) -> None:
        """
        Build the database: loads the documents, chunks them, and builds the vectors
        """
        me = self.__class__.__name__
        print(f"[{me}] Building database")
        if self.already_built() and not self.force_rebuild:
            print(f"[{me}] Database already built, loading from file")
            self.load_from_file()
        else:
            self.chunk_documents()
            self.build_vectors()
            self.build_vector_store()

    @abstractmethod
    def already_built(self) -> bool:
        """
        Check if the database is already built.
        Should be implemented in subclasses
        """

    @abstractmethod
    def load_from_file(self) -> None:
        """
        Load the database from a file.
        Should be implemented in subclasses
        """

    @abstractmethod
    def build_vectors(self) -> None:
        """
        Should write to self.vectors
        """

    @abstractmethod
    def build_vector_store(self) -> None:
        """
        Should implement the logic to create a vector store from the built vectors
        """
    
    @abstractmethod
    def query(self, query: str, top_k: int = 1) -> list[str]:
        """
        Query the database.
        """

    @abstractmethod
    def save_vectors(self, file_path: str) -> None:
        """
        Save the database to a file.
        """

    @abstractmethod
    def load_vectors_from_file(self, file_path: str) -> None:
        """
        Load the database from a file.
        Should have the same effect as build
        """


class EmbeddingsWithProgress:
    """Wrapper class that adds a progress bar to embeddings"""
    def __init__(self, embeddings : HuggingFaceEmbeddings, me : str) -> None:
        """
        Args:
            embeddings: HuggingFaceEmbeddings instance
            me: Name of the Database class using this wrapper
        """
        self.embeddings = embeddings
        self.me = me

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 10  # Adjust based on memory constraints
        embeddings = []

        with tqdm(
            total=len(texts), desc=f"[{self.me}] Embedding documents"
        ) as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embeddings.embed_query([text])


class FAISSDatabase(DataBase):
    """
    FAISS implementation of the RAG database using HuggingFace embeddings
    """
    
    def __init__(self, name: str, document_loaders: Iterator[RagDocumentLoader], embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2", persist_directory: str = None, **kwargs) -> None:
        """
        Initialize the FAISS database
        
        Args:
            name: Name of the database
            document_loaders: Iterator of document loaders
            embedding_model_name: Name of the HuggingFace model to use for embeddings(default: "sentence-transformers/all-mpnet-base-v2")
            persist_directory: Directory to persist the database to (default: {name}_faiss)
        """
        super().__init__(name, document_loaders, **kwargs)
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            show_progress=True,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
           # multi_process=True
        )
        self.persist_directory = persist_directory or f"{name}_faiss"
        
    def build_vectors(self) -> None:
        """
        Prepare data for vector storage
        """
        me = self.__class__.__name__
        print(f"[{me}] Preparing vectors")
        
        self.texts = [doc.page_content for doc in self.chunks]
        self.metadatas = [doc.metadata for doc in self.chunks]
        
    def build_vector_store(self) -> None:
        """
        Build a FAISS vector store from the texts and metadatas
        """
        me = self.__class__.__name__
        print(f"[{me}] Building vector store")
        

        # Use the wrapper for embeddings
        #embeddings_with_progress = EmbeddingsWithProgress(self.embeddings, me)

        # Build the vector store
        self.vector_store = FAISS.from_texts(
            texts=self.texts,
            embedding=self.embeddings,
            metadatas=self.metadatas
        )
        del self.embeddings  # Free memory
        print(f"[{me}] Vector store built with {len(self.texts)} texts")
        print(f"[{me}] Vector store size: {self.vector_store.index.ntotal} vectors")

        
    def query(self, query: str, top_k: int = 1) -> list[str]:
        """
        Query the database.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved document contents
        """
        me = self.__class__.__name__
        print(f"[{me}] Querying: {query}")
        
        results = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
        
    def save_vectors(self, file_path: str = None) -> None:
        """
        Save the FAISS database to the specified directory.
        If no directory is provided, use the default persist directory.
        
        Args:
            file_path: Path to save the database (default: self.persist_directory)
        """
        me = self.__class__.__name__
        path = file_path or self.persist_directory
        print(f"[{me}] Saving vectors to {path}")
        
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        
        # FAISS save_local method
        self.vector_store.save_local(path)
        
    def load_vectors_from_file(self, file_path: str = None) -> None:
        """
        Load the database from a file.
        
        Args:
            file_path: Path to load the database from (default: self.persist_directory)
        """
        me = self.__class__.__name__
        path = file_path or self.persist_directory
        print(f"[{me}] Loading vectors from {path}")
        
        self.vector_store = FAISS.load_local(
            folder_path=path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
    def already_built(self) -> bool:
        """
        Check if the database is already built by checking if the persist directory exists
        and contains FAISS index files
        
        Returns:
            True if the database is already built, False otherwise
        """
        if not os.path.exists(self.persist_directory) or not os.path.isdir(self.persist_directory):
            return False
        # Check for FAISS index file
        index_path = os.path.join(self.persist_directory, "index.faiss")
        return os.path.exists(index_path)
        
    def load_from_file(self) -> None:
        """
        Load the database from the default persist directory
        """
        self.load_vectors_from_file(self.persist_directory)


if __name__ == "__main__":
    # Example usage
    from .document_loaders import ReliawikiLoader, RagTextLoader
    genai_cheatsheet = RagTextLoader(
        "./ragdata/genaicheatsheet.txt",
        encoding="utf-8",
    )
    document_loaders = [genai_cheatsheet]
    database = FAISSDatabase("genai_cheatsheet", document_loaders)
    database.build()

    database.save_vectors()
    results = database.query("What is the reliability of a system?", top_k=3)
    print("Query results:\n" + "\n".join(f"{i+1}. {result}" for i, result in enumerate(results)))

    while True:
        query = input("Enter a query (type exit to quit): ")
        if query.lower() == "exit":
            break
        results = database.query(query, top_k=3)
        print("Query results:"+"\n".join(map(str, results)))
