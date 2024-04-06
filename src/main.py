import model
import prompter
import benchmarker


if __name__ == "__main__":
        
    croissant = model.HuggingFace("croissant", "croissantllm/CroissantLLMBase")
    croissant.model_init("../cache")


    prompteur = prompter.Prompter(
        croissant,
        "../data_set/json/example_problem.json",
        savepath="../data_set/json/completions.json",
    )


    benchmarkeur = benchmarker.Benchmarker(
        "../data_set/json/example_problem.json", "../data_set/json/completions.json"
    )

    benchmarkeur.benchmark()
    benchmarkeur.save_results()
