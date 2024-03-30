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
            self.gen_file = self.load_gen_file()
            pass
        else:
            self.gen_answers_with_model()

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
        f.close()
        return x
        
    def gen_answers_with_model(self):
        pass

    def benchmark(self):
        """_summary_
        """
        for i in tqdm.tqdm(range(20)):
            time.sleep(0.1)


if __name__ == '__main__':
    bm = Benchmarker('../data_set/json/example_problem.json', '../data_set/json/example_submission.json')
    bm.benchmark()