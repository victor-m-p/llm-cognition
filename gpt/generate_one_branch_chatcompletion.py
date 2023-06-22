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

lst=['One thing I like', '1']
lst[1] = 'hello'
lst
for i in lst: 
    if len(i.split()) > 1: 
        
responses_could = [s for s in responses_could if len(s.split()) > 1]

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
prompts = ['What is one thing that Heinz could do?',
           'What is one thing that Josh could do?',
           'What is one thing that Brian could do?',
           'What is one thing that Liz could do?',
           'What is one thing that Mary could do?',
           'What is one thing that Brad could do?']
contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]

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
        # when answer length = 1 it is typically a mistake;
        # i.e., "1" (where the model wants 1. ....). 
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

def generate_completions_sequential(vignettes, prompts, num_generations, temperature):
    contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
    generation_dict = {}
    for num, context in tqdm(enumerate(contexts)): 
        generation_list = []
        for _ in range(num_generations):
            completion = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "user", "content": context}],
              stop=['.', '?', '!', '\n\n'],
              temperature=temperature,
              n=1
              )
            text = completion['choices'][0]['message']['content']
            generation_list.append(text)
        generation_dict['context_' + str(num)] = {}
        generation_dict['context_' + str(num)]['vignette'] = vignettes[num]
        generation_dict['context_' + str(num)]['prompt'] = prompts[num]
        generation_dict['context_' + str(num)]['generation'] = generation_list
    return generation_dict

# run this for n=100 as a test 
'''
temperature_grid = [0.5, 1.0]
for temperature in temperature_grid: 
    generation_dict = generate_completions_at_once(vignettes, prompts, 100, temperature)
    with open(f'data/phillips2017_gpt-3.5-turbo_n100_temp{temperature}.json', 'w') as fp:
        json.dump(generation_dict, fp)
'''

# wtf ... this is ridiculous.
temperature_grid = [0.5, 1.0]
for temperature in temperature_grid: 
    generation_dict = generate_completions_sequential(vignettes, prompts, 100, temperature)
    with open(f'data/phillips2017_gpt-3.5-turbo_n100_temp{temperature}_sequential.json', 'w') as fp:
        json.dump(generation_dict, fp)