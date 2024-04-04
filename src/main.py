import model
import prompter



croissant = model.HuggingFace('croissant','croissantllm/CroissantLLMBase')
croissant.model_init('../cache')

prompteur = prompter.Prompter(croissant, '../data_set/json/example_problem.json')

prompteur.load_challenges_file('../data_set/json/example_problem.json')
prompteur.prompt()
prompteur.output_json()