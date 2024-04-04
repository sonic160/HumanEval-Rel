from model import Model
from sandbox_code_runner import bcolors
import json
import tqdm
from datetime import datetime
import sys


class Prompter:
    """_summary_
    """    
    def __init__(self, model: Model,challenges_file: str, savepath: str = None) -> None:
        """
        Initializes the Prompter with a given model, challenges file, and an optional save path.

        Args:
            model (Model): The language model to use for generating completions.
            challenges_file (str): The path to the JSON file containing the benchmark.
            savepath (str, optional): The path where the output JSON will be saved. Defaults to None.
        """     
        self.model = model
        self.savepath = savepath

        print('Loading the benchmark json file...')
        self.challenges = self.load_challenges_file(challenges_file)
        self.completions = []

        self.prompt()
        self.output_json()

    def load_challenges_file(self, chalfile: str) -> None:
        '''
        Loads the file that hosts the challenges that are to be asked to the LLM.

        Args:
                chalfile (str): The path to the file that hosts the challenges that are to be asked to the LLM.
        '''
        try:
            f = open(chalfile)

        except FileNotFoundError:
            print(bcolors.WARNING, f'File not found: {chalfile}. The location you gave is probably incorrect, or the file doesn\'t exist.', bcolors.ENDC)
            exit(1)

        try:
            x = json.load(f)

        except json.decoder.JSONDecodeError:
            print(bcolors.WARNING + 'The file you provided is badly formatted and/or is not JSON.' + bcolors.ENDC)
            exit(1)
        
        f.close()
        print('Loaded!\n')

        return x

    def prompt_batched(self) -> None:
        """
        Iterates over challenges, generates answers using the model, and stores them in the completions list.
        This version uses batching to generate answers more quickly.
        """    
        raise NotImplementedError

    def prompt(self) -> None:
        """
        Iterates over challenges, generates answers using the model, and stores them in the completions list.
        """     
        print("Prompting the LLM for answers...\n")

        for challenge in tqdm.tqdm(self.challenges):

            id, prompt = challenge['task_id'], challenge['prompt']
            completion = self.model.generate(prompt)
            self.completions.append({'task_id': id, 'completion': completion})

    def output_json(self) -> None:
        '''
        saves the LLM's answers to the benchmark questions as a json file

                Parameters:
                        chalfile (str): The path to the file that hosts the challenges that are to be asked to the LLM.
        '''
        if self.savepath == None:
            savepath =  str(f'../data_set/json/completions_{datetime.now()}_{self.model.model_name}.json').replace(' ', '_')
            self.savepath = savepath.replace(':', '_')
            
        
        print(self.savepath)
        json.dump(self.completions, open(self.savepath, 'w'))


def parse_args() -> tuple[bool, str, str]:
    pass

if __name__ == '__main__':
    #TODO: Add command line arguments
    print(sys.argv)
    model = Model('test')
    prompter = Prompter(model, '../data_set/json/example_problem.json')
