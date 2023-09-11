'''
VMP 2023-09-11:
This script is used to generate the data from GPT-4
'''

import openai 
import os 
from dotenv import load_dotenv
import json 
import re 
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time 

# setup 
outpath='../data/data_output/gpt4_new/'
inpath='../data/data_input/phillips2017.json'
model='gpt-4'
condition='could' # also run with 'should'
num_generations=10 
num_runs=10  
temperature=0.8
frequency=0.0
presence=0.0

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# quick function to ensure punctuation and space
def ensure_punctuation_and_space(s):
    if not s.endswith('.'):
        s = s + '.'
    return s + ' '

# load data
with open(inpath) as user_file:
  phillips2017 = json.load(user_file)
  
# wrangle data 
vignettes = [phillips2017[key]['vignette'] for key in phillips2017.keys()]
vignettes = [ensure_punctuation_and_space(s) for s in vignettes]
prompts = ['Please come up with 6 things that Heinz could do in this situation. Number them like this: 1. one thing that they could do is ... and so on. Always begin each number with the phrase "one thing that they could do is"',
           'Please come up with 6 things that Josh could do in this situation. Number them like this: 1. one thing that they could do is ... and so on. Always begin each number with the phrase "one thing that they could do is"',
           'Please come up with 6 things that Brian could do in this situation. Number them like this: 1. one thing that they could do is ... and so on. Always begin each number with the phrase "one thing that they could do is"',
           'Please come up with 6 things that Liz could do in this situation. Number them like this: 1. one thing that they could do is ... and so on. Always begin each number with the phrase "one thing that they could do is"',
           'Please come up with 6 things that Mary could do in this situation. Number them like this: 1. one thing that they could do is ... and so on. Always begin each number with the phrase "one thing that they could do is"',
           'Please come up with 6 things that Brad could do in this situation. Number them like this: 1. one thing that they could do is ... and so on. Always begin each number with the phrase "one thing that they could do is"']
contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
if condition == 'should':
    prompts = [re.sub('could', 'should', s) for s in prompts]

# function to create completion
@retry(wait=wait_random_exponential(min=1, max=200), stop=stop_after_attempt(10))
def create_completion(context, model, num_generations, 
                      temperature, frequency, presence):
    completion = openai.ChatCompletion.create(
          model=model,
          messages=[{"role": "user", "content": context}],
          max_tokens=600,
          temperature=temperature,
          frequency_penalty=frequency,
          presence_penalty=presence,
          n=num_generations
          )
    return completion 

# run generation over all contexts
ids = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad'] 
num_total=num_generations*num_runs
for num, prompt in enumerate(prompts):
    print(f'now running: {ids[num]}')
    context = vignettes[num] + prompt
    generation_list_outer = []
    generation_dict = {}
    id = ids[num]
    for i in range(num_runs):
        print(f'{i}')
        completion = create_completion(context, model, num_generations,
                                       temperature, frequency, presence)
        generation_list_inner = [completion['choices'][num]['message']['content'] for num in range(num_generations)]
        generation_list_outer.append(generation_list_inner)
        time.sleep(60)
    generation_list_flat = [item for sublist in generation_list_outer for item in sublist]
    generation_dict['context_' + str(num)] = {}
    generation_dict['context_' + str(num)]['id'] = id
    generation_dict['context_' + str(num)]['vignette'] = vignettes[num]
    generation_dict['context_' + str(num)]['prompt'] = prompts[num]
    generation_dict['context_' + str(num)]['generation'] = generation_list_flat
    path_string = f'{model}_n{num_total}_{id}_temp{temperature}_f{frequency}_p{presence}_{condition}.json'
    out_string = os.path.join(outpath, path_string)
    with open(out_string, 'w') as fp:
        json.dump(generation_dict, fp)