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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# load vignettes
with open('../huggingface/data_input/phillips2017.json') as user_file:
  phillips2017 = json.load(user_file)
phillips_context = [phillips2017[key]['vignette'] for key in phillips2017.keys()]

def ensure_punctuation_and_space(s):
    if not s.endswith('.'):
        s = s + '.'
    return s + ' '

import re

def get_first_sentence(s):
    match = re.search(r"([^.!?]*[.!?])", s)
    return match.group(0).strip() if match else ""

phillips_context = [ensure_punctuation_and_space(s) for s in phillips_context]

# create would and should condition
gender_pattern = [' he', ' he', ' he', ' she', ' she', ' he']
starting_pattern = 'One thing that'
condition_dict = {'could': ' could do is to', 'should': ' should do is to'}

# create inputs
all_contexts = []
for context, gender in zip(phillips_context, gender_pattern):
    starting_string = context + starting_pattern + gender 
    for key, val in condition_dict.items():
        full_string = starting_string + val
        all_contexts.append((key, full_string))

num_generations = 10
max_tokens = 50 # sentences typically < 30 tokens
results = {}
for pairs in all_contexts: 
    condition, prompt = pairs 
    generation_list = []
    for i in range(num_generations): 
        completion = openai.Completion.create(
          model="text-davinci-003",
          prompt=prompt,
          max_tokens=100, 
          temperature=1, 
          n=1,
          stop=['.', '?', '!']) # stop at first period 
        text = completion['choices'][0]['text']
        #text = get_first_sentence(text)
        generation_list.append(text)
    results[prompt] = {}
    results[prompt][condition] = generation_list

# issues: 
## 1. sometimes a sentence (e.g. with a comma) actually gives two answers

# save 
with open(f'data/text-davinci-003_phillips2017_n{num_generations}_could_should.json', 'w') as f:
  json.dump(results, f)