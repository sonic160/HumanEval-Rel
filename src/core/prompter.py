from core.sandbox_code_runner import bcolors
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime
import json
import sys
import tqdm
# make sure progress bars resize with terminal size
from functools import partial
tqdm.tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# Custom imports.
from core.system import System
from core.model import TestModel
import traceback

class Prompter:
    """
    The Prompter class is responsible for prompting a completion_system with a set of challenges,
    collecting the systems's responses, and saving them to a JSON file.

    The class is initialized with a completion_system, a path to a JSON file containing the challenges, and an optional save path.
    If no save path is provided, the responses will be saved in a default location.
    """

    def __init__(
        self,
        completion_system: System,
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
        show_example: bool=True,
        recovery_file: str=None,
        attempt_recovery: bool=True,
        parallel: bool=False,
    ) -> None:
        """
        Initializes the Prompter with a given model, challenges file, and an optional save path.

        Args:
            completion_system (System): The language model to use for generating completions.
            challenges_file (str): The path to the JSON file containing the benchmark.
            savepath (str, optional): The path where the output JSON will be saved. Defaults to None.
        """
        self.completion_system: System = completion_system
        self.savepath = savepath
        self.n = n
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.stream = stream
        self.show_example = show_example
        if recovery_file is not None:
            self.recovery_file = recovery_file
        elif savepath is not None:
            self.recovery_file = savepath.replace(".json", "_recovery.json")
        self.recovery_file = self.recovery_file.replace(" ", "_")
        
        print("Loading the benchmark json file...")
        self.challenges = self.load_challenges_file(challenges_file)
        self.completions = []

        # Check for recovery file
        if attempt_recovery and os.path.exists(self.recovery_file):
            self.load_recovery_state()
            print(f"Recovered {len(self.completions)} completions from {self.recovery_file}")

        if batch:
            self.prompt_batched(batch_size=batch_size)
        elif parallel:
            self.prompt_parrallel()
        else:
            self.prompt()
            
        # Delete recovery file after successful completion
        if os.path.exists(self.recovery_file):
            os.remove(self.recovery_file)

        self.output_json()

    def save_recovery_state(self):
        """
        Saves the current state of completions to a recovery file.
        """
        try:
            with open(self.recovery_file, 'w') as f:
                json.dump(self.completions, f)
        except Exception as e:
            print(f"Failed to save recovery state: {e}")

    def load_recovery_state(self):
        """
        Loads completions from a recovery file.
        """
        try:
            with open(self.recovery_file, 'r') as f:
                self.completions = json.load(f)
        except Exception as e:
            print(f"Failed to load recovery state: {e}")
            self.completions = []

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


        # Filter out challenges that have already been completed
        completed_task_ids = set(item['task_id'] for item in self.completions)
        unprocessed_challenges = [c for c in self.challenges if c['task_id'] not in completed_task_ids]

        batches = regroup(unprocessed_challenges, batch_size)
        for i, batch in enumerate(tqdm.tqdm(batches)):
            prompts = list(map(lambda l : l['prompt'], batch))
            task_ids = list(map(lambda l : l['task_id'], batch))
            try:
                completions = self.completion_system.generate_batch(prompts)
                for i in range(len(batch)):
                    if i < len(completions):  # Safety check
                        self.completions.append({'task_id': task_ids[i], 'completion': completions[i]})

                self.save_recovery_state()

            except Exception as e:
                print(f"error in batched generation for {self.completion_system.model_name}")
                print(e)

    def prompt(self) -> None:
        """
        Iterates over challenges, generates answers using the model, and stores them in the completions list.
        """
        print(f"Prompting {self.completion_system.model_name} for answers...\n")

        # Filter out challenges that have already been completed
        completed_task_ids = set(item['task_id'] for item in self.completions)
        unprocessed_challenges = [c for c in self.challenges if c['task_id'] not in completed_task_ids]

        for i, challenge in enumerate(tqdm.tqdm(unprocessed_challenges, position=0, desc="challenges")):
            id, challenge_prompt = challenge['task_id'], challenge['prompt']
            for _ in tqdm.tqdm(range(self.n), desc=f"collecting {self.n} samples for challenge{id}", position=1, leave=False):
                try:
                    completion, whole_answer = self.completion_system.generate(
                                prompt=challenge_prompt, 
                                max_tokens=self.max_tokens, 
                                top_p=0.95, 
                                top_k=60, 
                                temperature=self.temperature,
                                prompt_prefix=self.prompt_prefix,
                                prompt_suffix=self.prompt_suffix,
                                stream=self.stream,
                 )
                    self.completions.append({
                            'task_id': id, 
                            'completion': completion,
                            'whole_answer': whole_answer
                            })
                except Exception as e:
                    print(f'[WARNING]: task_id number {id} problem with generation')
                    traceback.print_exc()
                    self.completions.append({'task_id': id, 'completion': "#error in generation"})

            self.save_recovery_state()
            
    def prompt_parrallel(self) -> None:
        """
        This method is optimized for Models that rely on LLM apis to generate completions.
        """
        print(
            f"Prompting {self.completion_system.model_name} for answers...\n")

        # Filter out challenges that have already been completed
        completed_task_ids = set(item['task_id'] for item in self.completions)
        unprocessed_challenges = [
            c for c in self.challenges if c['task_id'] not in completed_task_ids]
        
        task_stack = []
        for challenge in unprocessed_challenges:
            challenge_id, challenge_prompt = challenge['task_id'], challenge['prompt']
            if not self.show_example:
                challenge_prompt = self.remove_prompt_examples(challenge_prompt)
            for _ in range(self.n):
                task_stack.append((challenge_id, challenge_prompt)) 
                
        with ThreadPoolExecutor() as executor:
            future_to_task = {}
            for challenge_id, challenge_prompt in tqdm.tqdm(task_stack, desc="Submitting tasks", position=0):
                future = executor.submit(
                    self.completion_system.generate,
                    prompt=challenge_prompt,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    prompt_prefix=self.prompt_prefix,
                    prompt_suffix=self.prompt_suffix,
                    stream=self.stream,
                )
                future_to_task[future] = challenge_id

            for future in tqdm.tqdm(as_completed(future_to_task), desc="Collecting results", position=1, total=len(future_to_task)):
                challenge_id = future_to_task[future]
                try:
                    completion, whole_answer = future.result()
                    self.completions.append({
                        'task_id': challenge_id,
                        'completion': completion,
                        'whole_answer': whole_answer
                    })
                    self.save_recovery_state()
                except Exception as e:
                    print(f'[WARNING]: problem with generation')
                    traceback.print_exc()
                    self.completions.append({'task_id': challenge_id, 'completion': "#error in generation"})

      

    def remove_prompt_examples(self, challenge_prompt):
        lines = challenge_prompt.split("\n")
        new_lines = list()
        for i in range(len(lines)):
            if any((lines[i].strip().startswith("Example"),
                           lines[i].strip().startswith(">>>"),
                           i > 0 and lines[i-1].strip().startswith(">>>"))):
                continue
            new_lines.append(lines[i])
        challenge_prompt = "\n".join(new_lines)
        return challenge_prompt
      
    def output_json(self) -> None:
        """
        Saves the LLM's answers to the benchmark questions as a json file

        Parameters:
            chalfile (str): The path to the file that hosts the challenges that are to be asked to the LLM.
        """
        if self.savepath is None:
            self.savepath = str(
                f"../data_set/json/completions_{datetime.now()}_{self.completion_system.model_name}.json"
            ).replace(" ", "_")

        print(f"Saving {self.completion_system.model_name} answers to", self.savepath)
        json.dump(self.completions, open(self.savepath, 'w'))


def parse_args() -> tuple[bool, str, str]:
    """
    Parses the command line arguments and returns them as a tuple.
    """
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
    prompter = Prompter(model, "../data_set/json/example_problem.json", batch=False, n=1, 
                        recovery_file=args.recovery if hasattr(args, 'recovery') else None)
