import argparse
import json
from openai import OpenAI
from benchmarker import Benchmarker
from core.model import Model


def retrieve_and_format_batch_results(batch_id: str, output_file: str) -> None:
    """
    Retrieve batch results from OpenAI and format them as locally inferred generations.

    Args:
        batch_id (str): The ID of the batch job.
        output_file (str): The path to the output file where formatted results will be saved.
    """
    client = OpenAI(api_key="INSERT API KEY")
    batch_job = client.batches.retrieve(batch_id)
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    raw_result_file_name =  output_file+"l" # jsonl format
    with open(raw_result_file_name, 'wb') as file:
        file.write(result)

    formatted_results = []
    with open(raw_result_file_name, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            task_id = json_object['custom_id'].split('-')[1]
            completion = json_object['response']["body"]["choices"][0]["message"]["content"]
            extracted_code = Model.extract_code(completion)
            formatted_results.append({'task_id': task_id, 'completion': extracted_code})

    with open(output_file, 'w') as f:
        json.dump(formatted_results, f)

def main():
    parser = argparse.ArgumentParser(description='Download batch results from OpenAI or run benchmark on them')
    parser.add_argument('--action', type=str, required=True, choices=['download', 'benchmark', 'both'],
                        help='Action to perform: download, benchmark, or both')
    parser.add_argument('--batchid', type=str, required=True, help='Batch ID to process')
    parser.add_argument('--modelname', type=str, required=True, help='Name of the model used for batch generation')
    
    args = parser.parse_args()
    batch_id = args.batchid
    output_file = f"./batches/results_{batch_id}_{args.modelname}.json"
    
    if args.action in ['download', 'both']:
        print("retrieving and formatting batch results...")
        retrieve_and_format_batch_results(batch_id, output_file)
    
    if args.action in ['benchmark', 'both']:
        print("running benchmark on formatted results...")
        bm = Benchmarker(
            chal_file="./data_set/benchmark.json",
            completion_file=output_file,
            timeout_warnings=True,
            model_name=args.modelname
        )
        bm.run_benchmark()  # Ajout de l'appel à run_benchmark qui manquait

if __name__ == "__main__":
    main()
