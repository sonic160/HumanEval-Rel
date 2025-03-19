from openai import OpenAI
import argparse
from datetime import datetime
import tqdm
import json
from functools import partial
import math
import sys
import os

# make sure progress bars resize with terminal size
tqdm.tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# Default parameters
DEFAULT_PARAMS = {
    "N": 50,
    "MODEL": "gpt-4o",
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 500,
    "TOP_P": 0.95,
    "TOP_K": 60
}

# Pricing data as of 2025 (dollars per 1M tokens)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o3-mini": {"input": 1.00, "output": 4.00}
}

def load_challenges_file(chalfile: str) -> dict:
    """Load challenges from JSON file."""
    try:
        with open(chalfile) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading challenges file: {e}")
        sys.exit(1)

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in a text."""
    # Rough approximation: 1 token ≈ 4 characters
    return math.ceil(len(text) / 4)

def create_tasks(challenges, params):
    """Create batch tasks from challenges."""
    tasks = []
    for challenge in challenges:
        id, prompt = challenge['task_id'], challenge['prompt']
        for index in range(params["N"]):
            tasks.append({
                "custom_id": f"task-{id}-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": params["MODEL"],
                    "top_p": params["TOP_P"],
                    "temperature": params["TEMPERATURE"],
                    "max_tokens": params["MAX_TOKENS"],
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                }
            })
    return tasks

def write_batch_file(tasks, params, batch_dir="batches"):
    """Write tasks to a JSONL batch file."""
    # Ensure batch directory exists
    os.makedirs(batch_dir, exist_ok=True)
    
    batch_filename = f"{batch_dir}/{params['MODEL']}-top_p{params['TOP_P']}-top_k{params['TOP_K']}-temp{params['TEMPERATURE']}-N{params['N']}.jsonl"
    try:
        with open(batch_filename, 'w') as batch_file:
            for obj in tasks:
                batch_file.write(json.dumps(obj) + '\n')
        return batch_filename
    except IOError as e:
        print(f"Error writing batch file: {e}")
        sys.exit(1)

def calculate_cost(tasks, params):
    """Calculate estimated cost for batch tasks."""
    selected_model = params["MODEL"]
    model_rates = MODEL_PRICING.get(selected_model, default=MODEL_PRICING["gpt-4"])
    
    # Convert from per 1M tokens to per 1K tokens
    input_rate = model_rates["input"] / 1000  # dollars per 1k tokens
    output_rate = model_rates["output"] / 1000  # dollars per 1k tokens

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for task in tasks:
        messages = task["body"]["messages"]
        system_text = messages[0]["content"]
        user_text = messages[1]["content"]
        prompt_tokens = estimate_tokens(system_text) + estimate_tokens(user_text)
        completion_tokens = params["MAX_TOKENS"]
        
        total_input_tokens += prompt_tokens
        total_output_tokens += completion_tokens
        
        input_cost = prompt_tokens / 1000 * input_rate
        output_cost = completion_tokens / 1000 * output_rate
        task_cost = input_cost + output_cost
        total_cost += task_cost

    return {
        "total_cost": total_cost,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "model": selected_model,
        "input_rate": model_rates["input"],
        "output_rate": model_rates["output"]
    }

def send_batch(api_key, batch_filename):
    """Send batch file to OpenAI API."""
    try:
        client = OpenAI(api_key=api_key)
        
        batch_input_file = client.files.create(
            file=open(batch_filename, "rb"),
            purpose="batch"
        )

        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        batch_job = client.batches.retrieve(batch_job.id)
        return batch_job
    except Exception as e:
        print(f"Error sending batch: {e}")
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='OpenAI API batch processing utility')
    parser.add_argument('action', 
                        choices=['calculate', 'send'], 
                        help='Action to perform: calculate (cost estimation) or send (batch submission)')
    parser.add_argument('--api-key', 
                        help='OpenAI API key')
    parser.add_argument('--challenges', 
                        default='./data_set/benchmark.json',
                        help='Path to the challenges file (default: ./data_set/benchmark.json)')
    parser.add_argument('--model', 
                        default=DEFAULT_PARAMS["MODEL"],
                        choices=list(MODEL_PRICING.keys()),
                        help=f'OpenAI model to use (default: {DEFAULT_PARAMS["MODEL"]})')
    parser.add_argument('--n', 
                        type=int,
                        default=DEFAULT_PARAMS["N"],
                        help=f'Number of repetitions per challenge (default: {DEFAULT_PARAMS["N"]})')
    parser.add_argument('--temperature',
                        type=float, default=DEFAULT_PARAMS["TEMPERATURE"],
                        help=f'Temperature parameter (default: {DEFAULT_PARAMS["TEMPERATURE"]})')
    parser.add_argument('--max-tokens', 
                        type=int, default=DEFAULT_PARAMS["MAX_TOKENS"],
                        help=f'Maximum number of tokens for completion (default: {DEFAULT_PARAMS["MAX_TOKENS"]})')
    parser.add_argument('--top-p', 
                        type=float, 
                        default=DEFAULT_PARAMS["TOP_P"],
                        help=f'Top_p parameter (default: {DEFAULT_PARAMS["TOP_P"]})')
    parser.add_argument('--top-k', 
                        type=int, 
                        default=DEFAULT_PARAMS["TOP_K"],
                        help=f'Top_k parameter (default: {DEFAULT_PARAMS["TOP_K"]})')
    parser.add_argument('--batch-file', 
                         help='Use an existing batch file instead of creating a new one')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Build parameters from arguments
    params = {
        "N": args.n,
        "MODEL": args.model,
        "TEMPERATURE": args.temperature,
        "MAX_TOKENS": args.max_tokens,
        "TOP_P": args.top_p,
        "TOP_K": args.top_k
    }

    if args.action == 'calculate':
        challenges = load_challenges_file(args.challenges)
        tasks = create_tasks(challenges, params)
        cost_info = calculate_cost(tasks, params)

        print(f"\nCost estimation for model {cost_info['model']}:")
        print(f"  - Input rate: ${cost_info['input_rate']:.2f} per million tokens")
        print(f"  - Output rate: ${cost_info['output_rate']:.2f} per million tokens")
        print(f"  - Total input tokens: {cost_info['input_tokens']:,}")
        print(f"  - Estimated output tokens: {cost_info['output_tokens']:,}")
        print(f"  - TOTAL ESTIMATED COST: ${cost_info['total_cost']:.4f}")

    elif args.action == 'send':
        if not args.api_key:
            print("Error: API key is required for sending batches.")
            print("Use --api-key to specify your OpenAI API key.")
            sys.exit(1)

        batch_filename = args.batch_file
        if not batch_filename:
            print("Creating new batch file...")
            challenges = load_challenges_file(args.challenges)
            tasks = create_tasks(challenges, params)
            batch_filename = write_batch_file(tasks, params)
            cost_info = calculate_cost(tasks, params)
            print(
                f"Total estimated cost for batch call: ${cost_info['total_cost']:.4f} (using 1 token ≈ 4 characters)"
            )
        else:
            print(f"Using existing batch file: {batch_filename}")
            existing_tasks = []
            with open(batch_filename, 'r') as f:
                for line in f:
                    existing_tasks.append(json.loads(line.strip()))
            cost_info = calculate_cost(existing_tasks, params)
            print(f"Total estimated cost for batch call: ${cost_info['total_cost']:.4f} (using 1 token ≈ 4 characters)")

        input("Are you sure you want to send the batch file? Press Enter to continue or Ctrl+C to cancel.")

        print("Sending batch file...")
        batch_job = send_batch(args.api_key, batch_filename)
        print("Batch job created with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"Batch job ID: {batch_job.id}")
        print(f"Status: {batch_job.status}")

if __name__ == "__main__":
    main()
