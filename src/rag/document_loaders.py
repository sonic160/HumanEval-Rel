import locale

# this fixes a locale error on some systems
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, "C")  
    except locale.Error:
        pass  

import pandas as pd
from typing import Iterator, Annotated
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import Union, Optional
from pathlib import Path


Separators = Annotated[list[str], "A set of chunking separators adapted to the document type"]
MARKDOWN_SEPARATORS : Separators = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

class RagDocumentLoader(BaseLoader):
    """Base class for document loaders used in RAG models.
       subclasses should define self.separators at __init__ 
    """
    separators: Separators


class ReliawikiLoader(RagDocumentLoader):
    """Loads documents from the Reliawiki dataset."""

    def __init__(self, path: Annotated[str, "Path to the Reliawiki dataset csv"]) -> None:
        self.path = path
        self.separators = MARKDOWN_SEPARATORS

    def lazy_load(self) -> Iterator[Document]: 
        # Load the dataset with pandas
        df = pd.read_csv(self.path)
        # Iterate over the rows and yield a document
        for _, row in df.iterrows():
            yield Document(page_content=row["content"], metadata={"title": row["title"], "source": "Reliawiki" })


class RagTextLoader(TextLoader, RagDocumentLoader):
    """Loads documents from a text file."""

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
        separators: Separators = MARKDOWN_SEPARATORS
    ) -> None:
        self.separators = separators
        super().__init__(file_path, encoding=encoding, autodetect_encoding=autodetect_encoding)
