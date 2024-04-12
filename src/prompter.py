from model import Model
from sandbox_code_runner import bcolors
import json
import tqdm
from datetime import datetime
import sys
import argparse


class Prompter:
    """
    The Prompter class is responsible for prompting a language model with a set of challenges,
    collecting the model's responses, and saving them to a JSON file.

    The class is initialized with a model, a path to a JSON file containing the challenges, and an optional save path.
    If no save path is provided, the responses will be saved in a default location.
    """

    def __init__(
        self, model: Model, challenges_file: str, savepath: str = None, batch: bool = False
    ) -> None:
        """
        Initializes the Prompter with a given model, challenges file, and an optional save path.

        Args:
            model (Model): The language model to use for generating completions.
            challenges_file (str): The path to the JSON file containing the benchmark.
            savepath (str, optional): The path where the output JSON will be saved. Defaults to None.
        """
        self.model = model
        self.savepath = savepath

        print("Loading the benchmark json file...")
        self.challenges = self.load_challenges_file(challenges_file)
        self.completions = []
        self.prompt() if not batch else self.prompt_batched()
        
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

    def prompt_batched(self, batch_size = 8) -> None:
        """
        Iterates over challenges, generates answers using the model, and stores them in the completions list.
        This version uses batching to generate answers more quickly.
        """
        def regroup(chalenge, size):
            output_list = []
            for i in range(0, len(chalenge), size):
                # Append sublist of size n to the output list
                output_list.append(chalenge[i:i + size])
    
            return output_list
        batches = regroup(self.challenges,batch_size)
        for batch in tqdm.tqdm(batches):
            prompts = list(map(lambda l : l["prompt"],batch))
            task_ids = list(map(lambda l : l["task_id"],batch))
            try:
                completions = self.model.generate_batch(prompts)
                for i in range(batch_size):
                    self.completions.append({"task_id": task_ids[i], "completion": completions[i]})
            except Exception as e:
                print("error")
                print(e)

    def prompt(self) -> None:
        """
        Iterates over challenges, generates answers using the model, and stores them in the completions list.
        """
        print("Prompting the LLM for answers...\n")

        for challenge in tqdm.tqdm(self.challenges):

            id, prompt = challenge["task_id"], challenge["prompt"]
            try:
                completion = self.model.generate(prompt)
                self.completions.append({"task_id": id, "completion": completion})
            except Exception as e:
                print(f'task_id number {id} problem with generation')
                print(e)
                self.completions.append({"task_id": id, "completion": ""})

                    
    def output_json(self) -> None:
        """
        Saves the LLM's answers to the benchmark questions as a json file

                Parameters:
                        chalfile (str): The path to the file that hosts the challenges that are to be asked to the LLM.
        """
        if self.savepath == None:
            savepath = str(
                f"../data_set/json/completions_{datetime.now()}_{self.model.model_name}.json"
            ).replace(" ", "_")
            self.savepath = savepath.replace(":", "_")

        print(self.savepath)
        json.dump(self.completions, open(self.savepath, "w"))
        print("Done!")


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


if __name__ == "__main__":
    # TODO: Add command line arguments
    model = Model("test")
    prompter = Prompter(model, "../data_set/json/example_problem.json")
    # args = parse_args()
    # print(args.source, args.dest)
