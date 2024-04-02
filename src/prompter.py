from model import Model
import json


class Prompter:
    def __init__(self, model,challenges_file) -> None:
        self.model = model
        self.challenges = self.load_challenges_file(challenges_file)
        self.completions = []

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

    def prompt_batched(self):
        pass

    def prompt(self):
        
        for challenge in self.challenges:
            id , prompt, tests, entry_point = challenge['task_id'], challenge['prompt'], \
                    challenge['test'], challenge['entry_point']
            completion = self.model.generate(prompt)
            self.completions.append({'task_id': challenge['task_id'], 'completion': completion})

    def output_json(self):
        json.dump(self.completions, open('../data_set/json/completions.json', 'w'))
