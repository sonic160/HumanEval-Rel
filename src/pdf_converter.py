import pdfplumber
import re
from pylatexenc.latexencode import unicode_to_latex
from datasets import Dataset
from datasets import load_dataset



class PdfConverter:

    def __init__(self, pdfs: list) -> None:
        self.documents = pdfs
        self.texts =[]
        pass
    def __is_mathematical_expression(string):
    # Function to determine if a string likely represents a mathematical expression
    # This uses Unicode blocks common for math symbols and structured formulae
        if re.search(r'^([-+/*]\d+(\.\d+)?)+', string):
            return True
        """
        for mapping in mappings:
            if mapping in string:
                return True
        """
        return False

    def __write_text_output(self, text: str) -> None:
        # This method writes the text to a file
        with open('Output.txt', 'w') as f:
            f.write(text)
        
    def __to_dataset(self) -> Dataset:
        # This method converts the text to a dataset
        return load_dataset("text", data_files={"train": 'Output.txt'})
    

    def __join_and_clean(self, pages: list) -> str:
        text = "\n".join([txt.replace('\n', ' ').strip() for txt in "\n".join(pages).split("\n\n")]).strip()
        return text
    
    def __extract_document_text(self, doc: str) -> None:
        with pdfplumber.open(doc) as pdf:
            all_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Map extracted text to LaTeX
                    page_text = []
                    lines = text.split('\n')
                    for line in lines:
                        # Check if line contains mathematical expressions
                        if self.__is_mathematical_expression(line):
                            # Process the line as a mathematical expression
                            latex_line = unicode_to_latex(repr(line))
                            page_text.append(latex_line + '\n')
                        else:
                            page_text.append(line + '\n')

                    all_text.append(''.join(page_text))
                
        return self.__join_and_clean(all_text)


    def to_fine_tuning_dataset(self) -> Dataset:
        # This method converts the pdfs to a dataset
        for doc in self.documents:
            self.texts.append(self.__extract_document_text(doc))
        #TODO: Finish Implementation
        pass