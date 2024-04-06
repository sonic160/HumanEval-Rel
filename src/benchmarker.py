import tqdm
import time
import json
from sandbox_code_runner import SandboxCodeRunner
import numpy as np
from collections import defaultdict, Counter
from score_calculator import ScoreCalculator, PassAtK
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

N_WORKERS = 4
class Benchmarker:
    """_summary_@Benchmarker"""

    def __init__(self, chal_file, completion_file=None):

        self.completion_file = completion_file
        self.chalfile = chal_file
        print('Loading the files...')
        self.challenges = self.load_challenges_file()
        self.sandbox = SandboxCodeRunner()
        self.tests = dict()
        
        self.generations = self.load_gen_file()

        print('Starting benchmark...')
        self.benchmark_linear()
        print('Saving results...')
        self.save_results()
        print('Done.')       

    def load_challenges_file(self):
        """_summary_

        Returns:
            _type_: _description_
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

        self.score_model()

    def benchmark_parallel(self):
        with tqdm.tqdm(total=len(self.challenges)) as pbar:
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = []
                completion_id = Counter()

                for challenge in self.challenges:
                    future = executor.submit(self.benchmark_worker, challenge, completion_id)
                    futures.append(future)
                
                for future in as_completed(futures):
                    pbar.update(1)
                    id, answers = future.result()
                    self.tests[id] = answers
                
        self.score_model()
    #TODO: tqdm        
    def benchmark_worker(self, challenge, completion_id):
        id = challenge["task_id"]
        answers = []
        for gen in self.generations[challenge["task_id"]]:
            completion = gen["completion"]
            args = (challenge["prompt"], completion, challenge["test"], challenge["entry_point"])
            result, completion = self.sandbox.run_tests(*args)
            
            completion_id[challenge["task_id"]] += 1
            answers.append((result, completion))
        return id, answers
    
    def score_model(self):
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

        for k in [1, 10, 100]:
            pass_per_challenge = np.array([PassAtK().calculate_score(int(n), int(c), k) for n, c in zip(num_samples_it, correct)])
            if (total >= k).all():
                print(f"pass@{k}: {pass_per_challenge.mean()}")
        
        

    def benchmark(self):
        """_summary_"""
        # TODO: Make this method shorter (refactor)
        n_workers = 4

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
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
                    future = executor.submit(self.sandbox.run_tests, *args)
                    futures.append(future)
                    completion_id[id] += 1

                for future in tqdm.tqdm(as_completed(futures), total=len(futures), position=1, leave=False):
                    result, completion = future.result()

                    if id not in self.tests:
                        self.tests[id] = [(result, completion)]
                    else:
                        self.tests[id].append((result, completion))

            self.score_model()

    def save_results(self):
        json.dump(self.tests, open("../data_set/json/results.json", "w"))


if __name__ == "__main__":
    bm = Benchmarker(
        "../data_set/json/big.json",
        "../data_set/json/completions.json",
    )
    #bm.benchmark_parallel()
    #bm.save_results()
