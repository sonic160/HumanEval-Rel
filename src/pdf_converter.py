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
    def __is_mathematical_expression(self, string):
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
            i = 0
            for page in pdf.pages:
                i+=1
                text = page.extract_text()
                if text:
               
                    # Map extracted text to LaTeX
                    paragraphs = text.split('\n\n')
                    for paragraph in paragraphs:
                        # Check if line contains mathematical expressions
                        if not self.__is_mathematical_expression(paragraph):
                            all_text.append(paragraph.replace("\n", " "))

        print(len(all_text))
        return all_text


    def to_fine_tuning_dataset(self) -> Dataset:
        # This method converts the pdfs to a dataset
        for doc in self.documents:
            for paragraph in self.__extract_document_text(doc):
                self.texts.append(paragraph)
        print("___")
        print(len(self.texts))
        data = {"text": self.texts}
        dataset = Dataset.from_dict(data)
        print(dataset)
        return self.texts


if __name__ == "__main__":
    pdfs = ['pdf reliability.pdf']
    pdf_converter = PdfConverter(pdfs)
    pdf_converter.to_fine_tuning_dataset()
    