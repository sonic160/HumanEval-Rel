import model
import prompter
import benchmarker

croissant = model.HuggingFace("croissant", "croissantllm/CroissantLLMBase")
croissant.model_init("../cache")

prompteur = prompter.Prompter(croissant, "../data_set/json/example_problem.json")

prompteur = prompter.Prompter(
    croissant,
    "../data_set/json/example_problem.json",
    savepath="../data_set/json/completions.json",
)

prompteur.load_challenges_file("../data_set/json/example_problem.json")
prompteur.prompt()
prompteur.output_json()

benchmarkeur = benchmarker.Benchmarker(
    "../data_set/json/example_problem.json", "../data_set/json/completions.json"
)
benchmarkeur.load_challenges_file()
benchmarkeur.load_gen_file()
benchmarkeur.benchmark()
benchmarkeur.save_results()
