'''
gpt-4
'''

import openai 
import os 
from dotenv import load_dotenv
import json 
from tqdm import tqdm 
import re 
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_incrementing
)
import time 

outpath='../data/data_output/gpt4/'
inpath='../data/data_input/phillips2017.json'
model='gpt-4'
condition='could' # chould
num_generations=5 # per time 
num_runs=10 # number of times 
temperature=0.8
frequency=0.8
presence=0.8

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ensure_punctuation_and_space(s):
    if not s.endswith('.'):
        s = s + '.'
    return s + ' '

with open(inpath) as user_file:
  phillips2017 = json.load(user_file)
  
vignettes = [phillips2017[key]['vignette'] for key in phillips2017.keys()]
vignettes = [ensure_punctuation_and_space(s) for s in vignettes]
prompts = ['Please come up with 20 things that Heinz could do in this situation. Number them like this: 1. One thing that Heinz could do is ... and so on. Always use "could"',
           'Please come up with 20 things that Josh could do in this situation. Number them like this: 1. One thing that Josh could do is ... and so on. Always use "could"',
           'Please come up with 20 things that Brian could do in this situation. Number them like this: 1. One thing that Brian could do is ... and so on. Always use "could"',
           'Please come up with 20 things that Liz could do in this situation. Number them like this: 1. One thing that Liz could do is ... and so on. Always use "could"',
           'Please come up with 20 things that Mary could do in this situation. Number them like this: 1. One thing that Mary could do is ... and so on. Always use "could"',
           'Please come up with 20 things that Brad could do in this situation. Number them like this: 1. One thing that Brad could do is ... and so on. Always use "could"']
contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
if condition == 'should':
    prompts = [re.sub('could', 'should', s) for s in prompts]

@retry(wait=wait_random_exponential(min=1, max=200), stop=stop_after_attempt(10))
def create_completion(context, model, num_generations, 
                      temperature, frequency, presence):
    completion = openai.ChatCompletion.create(
          model=model,
          messages=[{"role": "user", "content": context}],
          max_tokens=1500,
          temperature=temperature,
          frequency_penalty=frequency,
          presence_penalty=presence,
          n=num_generations
          )
    return completion 

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
        # make sure that each completions is at least two words
        # this should never happen now 
        #for j, ele in enumerate(generation_list_inner):
        #    if len(ele.split()) < 2:
        #        print('less than 2 words')
        #        length=1
        #        while length < 2:
        #            print('enter while')
        #            completion_fix = create_completion(context, model, 1,
        #                                               temperature, frequency, presence)
        #            text=generation_list_inner[i]=completion_fix['choices'][0]['message']['content']
        #            generation_list_inner[i]=text
        #            length = len(text.split())
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