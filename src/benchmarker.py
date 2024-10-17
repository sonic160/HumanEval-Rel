import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import numpy as np
import timeit
import tqdm
from typing import Tuple

# Custom imports.
from sandbox_code_runner import SandboxCodeRunner
from score_calculator import PassAtK

N_WORKERS = 20

class Benchmarker:
    """
    The Benchmarker class is responsible for benchmarking a language model's performance on a set of challenges.
    It loads the challenges and the model's responses from JSON files, runs tests on the responses, scores the model,
    and saves the results to a JSON file.
    """

    def __init__(self, chal_file, completion_file=None):
        """
        Initializes the Benchmarker with a challenges file and an optional completions file.

        Args:
            chal_file (str): The path to the JSON file containing the challenges.
            completion_file (str, optional): The path to the JSON file containing the model's responses. Defaults to None.
        """

        self.completion_file = completion_file
        self.chalfile = chal_file
        self.sandbox = SandboxCodeRunner()
        self.tests = dict()

        print("Loading the files...")
        self.challenges = self.load_challenges_file()
        self.generations = self.load_gen_file()

        print("Starting benchmark...")
        self.result = self.benchmark_linear()

        print("Saving results...")
        self.save_results()
        print("Done.")

    def load_challenges_file(self):
        """
        Loads the challenges from a JSON file.

        Returns:
            dict: A dictionary containing the challenges.
        """
        try:
            f = open(self.chalfile)
        except FileNotFoundError:
            print("File not found")
            exit(1)

        x = json.load(f)
        f.close()
        return x

    def load_gen_file(self):
        """
        Loads the model's responses from a JSON file.

        Returns:
            dict: A dictionary containing the model's responses.
        """
        try:
            f = open(self.completion_file)
        except FileNotFoundError:
            print("File not found)))")
            exit(1)

        x = json.load(f)
        gens = dict()
        for gen in x:
            if gen["task_id"] not in gens:
                gens[gen["task_id"]] = [gen]
            else:
                gens[gen["task_id"]].append(gen)

        f.close()
        return gens

    def benchmark_linear(self):
        """
        Runs tests on the model's responses in a linear manner, scores the model, and stores the results.
        """
        for challenge in tqdm.tqdm(self.challenges):
            id, prompt, tests, entry_point = (
                challenge["task_id"],
                challenge["prompt"],
                challenge["test"],
                challenge["entry_point"],
            )

            for gen in self.generations[id]:
                completion = gen["completion"]
                args = (prompt, completion, tests, entry_point)
                result, completion = self.sandbox.run_tests(*args)

                if id not in self.tests:
                    self.tests[id] = [(result, completion)]
                else:
                    self.tests[id].append((result, completion))
        return self.score_model()

    def benchmark_parallel(self):
        """
        Runs tests on the model's responses in parallel, scores the model, and stores the results.
        """
        with tqdm.tqdm(total=len(self.challenges)) as pbar:
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = []
                completion_id = Counter()

                for challenge in self.challenges:
                    future = executor.submit(
                        self.benchmark_worker, challenge, completion_id
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    pbar.update(1)
                    id, answers = future.result()
                    self.tests[id] = answers

        self.score_model()

    def benchmark_worker(self, challenge, completion_id):
        """
        A worker function for running tests on the model's responses in parallel.

        Args:
            challenge (dict): A dictionary containing a single challenge.
            completion_id (Counter): A counter for tracking the number of completions for each challenge.

        Returns:
            tuple: A tuple containing the challenge ID and a list of results.
        """
        id = challenge["task_id"]
        answers = []
        for gen in self.generations[challenge["task_id"]]:
            completion = gen["completion"]
            args = (
                challenge["prompt"],
                completion,
                challenge["test"],
                challenge["entry_point"],
            )
            result, completion = self.sandbox.run_tests(*args)

            completion_id[challenge["task_id"]] += 1
            answers.append((result, completion))
        return id, answers

    def score_model(self):
        """
        Scores the model based on the results of the tests.
        """
        total, correct = [], []

        for result in self.tests.values():
            result.sort()
            passed = [r[0] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))

        total = np.array(total)
        correct = np.array(correct)

        if isinstance(total, int):
            num_samples_it = itertools.repeat(total, len(correct))
        else:
            assert len(total) == len(correct)
            num_samples_it = iter(total)

        ks = [1, 5, 10, 100]
        pass_per_k = {k: None for k in ks}
        
        for k in ks:
            pass_per_challenge = np.array(
                [
                    PassAtK().calculate_score(int(n), int(c), k)
                    for n, c in zip(total, correct)
                ]
            )
            
            if (total >= k).all():
                print(f"pass@{k}: {pass_per_challenge.mean()}")
                pass_per_k[k] = pass_per_challenge.mean()

        return pass_per_k

    def save_results(self):
        """
        Saves the results of the tests to a JSON file.
        """
        json.dump(self.tests, open("./data_set/json/results.json", "w"))


def parse_args() -> tuple[bool, str, str]:
    """_summary_

    Returns:
        _type_: _description_
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
        "-c",
        "--completions",
        required=False,
        help="The path to the file where the completions JSON is saved;",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        required=False,
        action="store_true",
        help="Use parallel processing to perform the tests more quickly.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        required=False,
        type=int,
        help="Number of workers to use in parallel processing. Default is 20.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    bm = Benchmarker(
        "../data_set/json/prompt_file.json",
        "../data_set/json/completions_meta-llamaCodeLlama-34b-hf.json",
    )
    # arg_parser = parse_args()
    # print(arg_parser.source, arg_parser.completions, arg_parser.parallel)
    # print('\n\n Linear: '+str(timeit.timeit(bm.benchmark_linear, number=25))+'\n\n')
    # print('\n\n Parallel: '+str(timeit.timeit(bm.benchmark_parallel, number=25))+'\n\n')
    # bm.benchmark_parallel()
    # bm.save_results()
