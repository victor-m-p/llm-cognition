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

def generate_completions_batched(vignettes, prompts, num_generations, temperature):
    contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
    generation_dict = {}
    for num, context in tqdm(enumerate(contexts)): 
        completion = openai.Completion.create(
          model="text-davinci-003",
          prompt=context,
          stop=['.', '?', '!', '\n\n'],
          temperature=temperature,
          n=num_generations
          )
        generation_list = [completion['choices'][num]['text'] for num in range(num_generations)]
        generation_dict['context_' + str(num)] = {}
        generation_dict['context_' + str(num)]['vignette'] = vignettes[num]
        generation_dict['context_' + str(num)]['prompt'] = prompts[num]
        generation_dict['context_' + str(num)]['generation'] = generation_list
    return generation_dict

def generate_completions_sequential(vignettes, prompts, num_generations, temperature):
    contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
    generation_dict = {}
    for num, context in tqdm(enumerate(contexts)): 
        generation_list = []
        for _ in range(num_generations):
            completion = openai.Completion.create(
              model="text-davinci-003",
              prompt=context,
              stop=['.', '?', '!', '\n\n'],
              temperature=temperature,
              n=1
              )
            text = completion['choices'][0]['text']
            generation_list.append(text)
        generation_dict['context_' + str(num)] = {}
        generation_dict['context_' + str(num)]['vignette'] = vignettes[num]
        generation_dict['context_' + str(num)]['prompt'] = prompts[num]
        generation_dict['context_' + str(num)]['generation'] = generation_list
    return generation_dict

temperature_grid = [0.5, 1.0]
for temperature in temperature_grid: 
    generation_dict = generate_completions_batched(vignettes, prompts, 100, temperature)
    with open(f'data/phillips2017_text-davinci-003_n100_temp{temperature}_batched.json', 'w') as fp:
        json.dump(generation_dict, fp)