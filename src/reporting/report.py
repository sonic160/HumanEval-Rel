import os 
import json

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    result_file_path : str
    size : float = None #in billion parameters
    is_open_source : bool = True
    __data : dict = None


    @property
    def results(self) -> None:
        if self.__data is None:
            self.__load_data()
        return self.__data
    
    def __load_data(self) -> None:
        with open(self.result_file_path) as f:
            self.__data = json.load(f)

    

class Report(ABC):
    """
    The purpose of a report subclass is to produce a file that displays the results of an experiment
    """


    def __init__(self, models : list[Model]) -> None:
        self.models = models

    
    @abstractmethod
    def generate_report(self) -> None:
        """
        This method should produce a file 
        """
        raise NotImplementedError


ks = [1, 5, 10, 15]+[k*10 for k in range(2, 10+1)]

class OneTempReport(Report):
    """generates a markdown file with graphs and tables"""

    def __init__(self, results_files: list[str]) -> None:
        super().__init__(results_files)
        self.md = "# Pass@K Benchmark"


    def add(self, line) -> None:
        self.md += line + "\n"


    def generate_report(self) -> None:
        self.add_pass_per_k()
        with open(f"./reports/{self.get_report_number()}", "w") as f:
            f.write(self.md)

    def add_pass_per_k(self,) -> None:

        self.add("|Model Name|"+ "|".join(map(str, ks)) + "|" + "\n")
        self.add("|---------|"+"-|-".join("" for _ in range(len(ks)))+"-------------|")
        for model in self.models:
            self.add("\n")
            self.add("|"+model.name+"|")
            self.add("|".join(map(str, model.results)))
        
        

    def get_report_number(self) -> int:
        report_dir = "./reports"
        file_pattern = "report"
        file_extension = ".md"
        return 1+ sum(1 for file in os.listdir(report_dir) 
                if file.startswith(file_pattern) and file.endswith(file_extension))




if __name__ == "__main__":
    models_data = [
        {"name": "croissantllm/CroissantLLMBase", "size": 1.3, "is_open_source": True},
        {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "size": 7, "is_open_source": True},
        {"name": "mistralai/Ministral-8B-Instruct-2410", "size": 8, "is_open_source": True},
        {"name": "meta-llama/Llama-Guard-3-8B", "size": 8, "is_open_source": True},
        {"name": "princeton-nlp/gemma-2-9b-it-SimPO", "size": 9, "is_open_source": True},
        {"name": "google/gemma-2-9b-it", "size": 9, "is_open_source": True},
        {"name": "microsoft/Orca-2-13b", "size": 13, "is_open_source": True},
        {"name": "google/gemma-2-27b-it", "size": 27, "is_open_source": True},
        {"name": "nvidia/Llama-3.1-Nemotron-70B-Reward-HF", "size": 70, "is_open_source": True},
        {"name": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", "size": 70, "is_open_source": True},
        {"name": "mistralai/Mistral-Large-Instruct-2407", "size": 123, "is_open_source": True},
        {"name": "openai-community/roberta-large-openai-detector", "size": 355, "is_open_source": True},
        {"name": "deepseek-ai/DeepSeek-V2.5", "size": 2.5, "is_open_source": True},
    ]

    models = [
        Model(
            name=model_data["name"],
            result_file_path=f"./generated_data/experiment_results/results_{model_data['name'].replace('/', '')}.json",
            size=model_data["size"],
            is_open_source=model_data["is_open_source"]
        )
        for model_data in models_data
    ]