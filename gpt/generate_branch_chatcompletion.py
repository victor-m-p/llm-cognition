'''
Not included in original submission.
This is a script to generate completions for the Phillips et al. (2017) vignettes.
Using ChagGPT (gpt-3.5-turbo) rather than GPT3 (text-davinci-003). 
'''

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

outpath='../data/data_output/phillips_chatgpt/'
inpath='../data/data_input/phillips2017.json'
num_generations=100
num_times=10
condition='could' # should

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
prompts = ['What is one thing that Heinz could do?',
           'What is one thing that Josh could do?',
           'What is one thing that Brian could do?',
           'What is one thing that Liz could do?',
           'What is one thing that Mary could do?',
           'What is one thing that Brad could do?']
contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
if condition == 'should':
    prompts = [re.sub('could', 'should', s) for s in prompts]

def generate_completions_batched(vignettes, prompts, num_generations, temperature):
    contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
    generation_dict = {}
    for num, context in tqdm(enumerate(contexts)): 
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[{"role": "user", "content": context}],
          stop=['.', '?', '!', '\n\n'],
          temperature=temperature,
          n=num_generations
          )
        # add to generation list
        generation_list = [completion['choices'][num]['message']['content'] for num in range(num_generations)]
        
        # ensuring responsese at least length 2
        # i.e., some responses start with e.g. "1. " or newline. 
        # regenerate these responses. 
        for i in generation_list: 
            if len(i.split()) > 1: 
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": context}],
                    stop=['.', '?', '!', '\n\n'],
                    temperature=temperature,
                    n=1)
                generation_list[i]=completion['choices'][0]['message']['content']
                
        generation_dict['context_' + str(num)] = {}
        generation_dict['context_' + str(num)]['vignette'] = vignettes[num]
        generation_dict['context_' + str(num)]['prompt'] = prompts[num]
        generation_dict['context_' + str(num)]['generation'] = generation_list
    return generation_dict

temperature = 1 # [0.5, 1.0]
for i in range(num_times):
    #for temperature in temperature_grid: 
    generation_dict = generate_completions_batched(vignettes, prompts, num_generations, temperature)
    path_string = f'phillips2017_gpt-3.5-turbo_n{num_generations}_m{i}_temp{temperature}_{condition}_fix.json'
    with open(f'data/phillips2017_gpt-3.5-turbo_n100_temp{temperature}_sequential.json', 'w') as fp:
        json.dump(generation_dict, fp)