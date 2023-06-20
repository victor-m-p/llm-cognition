# still extremely constrained
import openai 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import requests 
from dotenv import load_dotenv
import json 
from tqdm import tqdm 
import re 

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ensure_punctuation_and_space(s):
    if not s.endswith('.'):
        s = s + '.'
    return s + ' '

with open('../huggingface/data_input/phillips2017.json') as user_file:
  phillips2017 = json.load(user_file)
  
vignettes = [phillips2017[key]['vignette'] for key in phillips2017.keys()]
vignettes = [ensure_punctuation_and_space(s) for s in vignettes]
prompts = ['One thing that Heinz could do is to',
           'One thing that Josh could do is to',
           'One thing that Brian could do is to',
           'One thing that Liz could do is to',
           'One thing that Mary could do is to',
           'One thing that Brad could do is to']

num_generations = 4
temperature = 1 # we should probably try with 0.5 (which I think is default)

def generate_completions(vignettes, prompts, num_generations, temperature):
    contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
    generation_dict = {}
    for num, context in tqdm(enumerate(contexts)): 
        generation_list = []
        for _ in range(num_generations):
            completion = openai.Completion.create(
                model="text-davinci-003",
                prompt=context,
                max_tokens=200, 
                temperature=temperature, 
                n=1, # seem more similar when generated at once so generating one at a time
                stop=['.', '?', '!', '\n\n']) 
            text = completion['choices'][0]['text']
            generation_list.append(text)
        generation_dict['context_' + str(num)] = {}
        generation_dict['context_' + str(num)]['vignette'] = vignettes[num]
        generation_dict['context_' + str(num)]['prompt'] = prompts[num]
        generation_dict['context_' + str(num)]['generation'] = generation_list
    return generation_dict

completion = openai.Completion.create(
    model='text-davinci-003',
    prompt='I',
    max_tokens=200,
    temperature=1,
    n=3,
    stop=['.', '?', '!', '\n\n']) 



generation_dict_05 = generate_completions(vignettes, prompts, num_generations, 0.5)


# issues: 
## 1. sometimes a sentence (e.g. with a comma) actually gives two answers

# save 
with open(f'data/text-davinci-003_phillips2017_mary_branches.json', 'w') as f:
  json.dump(list_sentence_context, f)