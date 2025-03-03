from openai import OpenAI
client = OpenAI(
     api_key="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"
)

from datetime import datetime
import tqdm
# make sure progress bars resize with terminal size
import json
from functools import partial
tqdm.tqdm = partial(tqdm.tqdm, dynamic_ncols=True)



def load_challenges_file(chalfile: str) -> dict:
    with open(chalfile) as f:
        f = open(chalfile)
        x = json.load(f)
        return x
    

# we use batch api to reduce the cost, also
# we can't use the usual workflow because we have to send all questions at once
#### GEN SETTINGS
PARAMS = {
    "N": 50,
    "MODEL": "gpt-4o",
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 500,
    "TOP_P": 0.95,
    "TOP_K": 60
}


challenges_file =  './data_set/json/prompt_file.json'

#create the batch file
tasks = []
challenges = load_challenges_file(challenges_file)
for challenge in challenges:
        id, prompt = challenge['task_id'], challenge['prompt']
        for index in range(PARAMS["N"]):
                tasks.append(
                        {
                                "custom_id": f"task-{id}-{index}",
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": {
                                        "model": PARAMS["MODEL"],
                                        "top_p": PARAMS["TOP_P"],
                                        "temperature": PARAMS["TEMPERATURE"],
                                        "max_tokens": PARAMS["MAX_TOKENS"],
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
                        }
                )
                
batch_filename = f"batches/{PARAMS['MODEL']}-top_p{PARAMS['TOP_P']}-top_k{PARAMS['TOP_K']}-temp{PARAMS['TEMPERATURE']}-N{PARAMS['N']}.jsonl"

with open(batch_filename, 'w') as batch_file:
    for obj in tasks:
        batch_file.write(json.dumps(obj) + '\n')


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
print("created batch job with parameters\n", PARAMS, "\nbatch_job id is ", batch_job.id)