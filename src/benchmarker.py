import tqdm
import time
import json 
from sandbox_code_runner import SandboxCodeRunner

class Benchmarker:
    """_summary_@Benchmarker        
    """
    def __init__(self, chal_file, completion_file=None):

        self.completion_file = completion_file
        self.chalfile = chal_file
        self.challenges = self.load_challenges_file()
        self.sandbox = SandboxCodeRunner()
        self.tests = dict()
        

        if self.challenges:
            self.generations = self.load_gen_file()
        else:
            self.generations = self.gen_answers_with_model()

        

    

    def load_challenges_file(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        try:
            f = open(self.chalfile)
        except FileNotFoundError:
            print('File not found')
            exit(1)

        x = json.load(f)
        print(x)
        f.close()
        return x
    
    def load_gen_file(self): 
        #TODO: Throw Exception if srcfile doesn't exist
        #with open(self.srcfile, 'r') as f:
            #return f.read()
        try:
            f = open(self.completion_file)
        except FileNotFoundError:
            print('File not found)))')
            exit(1)
        
        x = json.load(f)
        gens = dict()

        for gen in x:
            if gen['task_id'] not in gens:
                gens[gen['task_id']] = [gen]
            else:
                gens[gen['task_id']].append(gen)

        f.close()
        return gens
        
    def gen_answers_with_model(self):
        pass

    def benchmark(self):
        """_summary_
        """
        for i in tqdm.tqdm(range(len(self.challenges))):

            id  = self.challenges[i]['task_id']
            prompt = self.challenges[i]['prompt']
            tests = self.challenges[i]['test']
            entry_point = self.challenges[i]['entry_point']

            for gen in self.generations[id]:
                completion = gen['completion']
                result = self.sandbox.run_tests(prompt, completion, tests, entry_point)
                
                if tests not in self.tests:
                    self.tests[id] = [{'result': result, 'completion': completion}]
                else:
                    self.tests[id].append({'result': result, 'completion': completion})
    
    def save_results(self):
        json.dump(self.tests, open('../data_set/json/results.json', 'w'))

if __name__ == '__main__':
    bm = Benchmarker('../data_set/json/example_problem.json', '../data_set/json/example_submission.json')
    bm.benchmark()
    bm.save_results()