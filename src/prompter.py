import argparse
from datetime import datetime
import json
import sys
import tqdm
# make sure progress bars resize with terminal size
from functools import partial
tqdm.tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# Custom imports.
from model import Model, TestModel
from sandbox_code_runner import bcolors

class Prompter:
    """
    The Prompter class is responsible for prompting a language model with a set of challenges,
    collecting the model's responses, and saving them to a JSON file.

    The class is initialized with a model, a path to a JSON file containing the challenges, and an optional save path.
    If no save path is provided, the responses will be saved in a default location.
    """

    def __init__(
        self,
        model: Model,
        challenges_file: str,
        savepath: str=None,
        batch: bool=False,
        batch_size: int=8,
        temperature: float=0.3,
        max_tokens: int=1000,
        top_p: float=0.95,
        top_k: int=60,
        prompt_prefix: str="",
        prompt_suffix: str="",
        stream: bool=False,#show output of LLM in real time
        n: int=1,
    ) -> None:
        """
        Initializes the Prompter with a given model, challenges file, and an optional save path.

        Args:
            model (Model): The language model to use for generating completions.
            challenges_file (str): The path to the JSON file containing the benchmark.
            savepath (str, optional): The path where the output JSON will be saved. Defaults to None.
        """
        self.model : Model = model
        self.savepath = savepath
        self.n = n
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.stream = stream

        print("Loading the benchmark json file...")
        self.challenges = self.load_challenges_file(challenges_file)
        self.completions = []
        self.prompt() if not batch else self.prompt_batched(batch_size=batch_size)
        
        self.output_json()

    def load_challenges_file(self, chalfile: str) -> None:
        """
        Loads the file that hosts the challenges that are to be asked to the LLM.

        Args:
            chalfile (str): The path to the file that hosts the challenges that are to be asked to the LLM.
        """
        try:
            f = open(chalfile)

        except FileNotFoundError:
            print(
                bcolors.WARNING,
                f"File not found: {chalfile}. The location you gave is probably incorrect, or the file doesn't exist.",
                bcolors.ENDC,
            )
            exit(1)

        try:
            x = json.load(f)

        except json.decoder.JSONDecodeError:
            print(
                bcolors.WARNING
                + "The file you provided is badly formatted and/or is not JSON."
                + bcolors.ENDC
            )
            exit(1)

        f.close()
        print("Loaded!\n")
        return x

    def prompt_batched(self, batch_size=8) -> None:
        """
        Iterates over challenges, generates answers using the model, and stores them in the completions list.
        This version uses batching to generate answers more quickly.
        """
        def regroup(challenge, size):
            output_list = []
            for i in range(0, len(challenge), size):
                # Append sublist of size n to the output list
                for _ in range(self.n):
                    output_list.append(challenge[i:i + size])
            return output_list
        
        batches = regroup(self.challenges, batch_size)
        for batch in tqdm.tqdm(batches):
            prompts = list(map(lambda l : l['prompt'], batch))
            task_ids = list(map(lambda l : l['task_id'], batch))
            try:
                completions = self.model.generate_batch(prompts)
                for i in range(batch_size):
                    self.completions.append({'task_id': task_ids[i], 'completion': completions[i]})
            except Exception as e:
                print(f"error in batched genration for {self.model.name}")
                print(e)

    def prompt(self) -> None:
        """
        Iterates over challenges, generates answers using the model, and stores them in the completions list.
        """
        print(f"Prompting {self.model.model_name} for answers...\n")
        for challenge in tqdm.tqdm(self.challenges, position=0, desc="challenges"):
            id, challenge_prompt = challenge['task_id'], challenge['prompt']
            for _ in tqdm.tqdm(range(self.n), desc=f"collecting {self.n} samples for challenge{id}", position=1, leave=False):
                try:
                    completion, whole_answer = self.model.generate(
                                prompt=challenge_prompt, 
                                max_tokens=self.max_tokens, 
                                top_p=0.95, 
                                top_k=60, 
                                temperature=self.temperature,
                                prompt_prefix=self.prompt_prefix,
                                prompt_suffix=self.prompt_suffix,
                                stream=self.stream
                 )
                    self.completions.append({
                            'task_id': id, 
                            'completion': completion,
                            'whole_answer': whole_answer
                            })
                except Exception as e:
                    print(f'[WARNING]: task_id number {id} problem with generation')
                    print(e)
                    self.completions.append({'task_id': id, 'completion': "#error in generation"})
                    
    def output_json(self) -> None:
        """
        Saves the LLM's answers to the benchmark questions as a json file

        Parameters:
            chalfile (str): The path to the file that hosts the challenges that are to be asked to the LLM.
        """
        if self.savepath is None:
            savepath = str(
                f"../data_set/json/completions_{datetime.now()}_{self.model.model_name}.json"
            ).replace(" ", "_")
            self.savepath = savepath.replace(":", "_")

        print(f"Saving {self.model.model_name} answers to",self.savepath)
        json.dump(self.completions, open(self.savepath, 'w'))


def parse_args() -> tuple[bool, str, str]:
    """
    Parses the command line arguments and returns them as a tuple.
    """
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Prompt a language model with a set of challenges and save the completions to a JSON file."
    )
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        help="The path to the JSON file containing the challenges.",
    )
    parser.add_argument(
        "-d",
        "--dest",
        required=False,
        help="The path where the output JSON will be saved.",
    )
    parser.add_argument(
        "-b",
        "--batched",
        required=False,
        help="Use batched prompting to generate completions more quickly.",
    )

    return parser.parse_args()


if __name__ == '__main__':
    model = TestModel("test")
    prompter = Prompter(model, "../data_set/json/example_problem.json", batch=False, n=1)