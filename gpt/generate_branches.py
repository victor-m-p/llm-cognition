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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ensure_punctuation_and_space(s):
    if not s.endswith('.'):
        s = s + '.'
    return s + ' '

import re

def get_first_sentence(s):
    match = re.search(r"([^.!?]*[.!?])", s)
    return match.group(0).strip() if match else ""

#background = 'Josh is on the way to the airport to catch a flight for a hunting safari in Africa. He leaves with plenty of time to make it there, but his car breaks down on the highway. Now Josh is sitting in his car near a busy intersection, and knows he needs to get to airport soon if he is going to catch his flight.'
#prompt = ' One thing Josh could do is to'
#answer = ' reschedule for a later flight.'
background = "Mary is about to go to her final class of the day when she remebers that there is a homework assignment that is due. Mary's mom accidentally took her homework assignment out of her backpack when she was making room for her lunch. Now Mary has nothing to turn in for credit."
prompt = ' One thing that Mary could do is to'
answer = ' run home to get her homework.'
context = background + prompt

reg = r"(\s*\w+|\s*\b'\w*|\s*[.,?!:;])"
sentence_list=re.findall(reg, answer)
num_per_split = 10

list_without_context = []
list_full_context = []
list_sentence_context = []
sentence = ""
for i in tqdm(sentence_list): 
    sublist_without_context = []
    sublist_full_context = []
    sublist_sentence_context = []
    for j in range(num_per_split):
        completion = openai.Completion.create(
            model="text-davinci-003",
            prompt=context,
            max_tokens=100, 
            temperature=1, 
            n=1,
            stop=['.', '?', '!']) # stop at first period  
        text = completion['choices'][0]['text']
        sublist_without_context.append(text)
        sublist_full_context.append(context + text)
        sublist_sentence_context.append(prompt + sentence + text)
    list_without_context.append(sublist_without_context)
    list_full_context.append(sublist_full_context)
    list_sentence_context.append(sublist_sentence_context)
    sentence += i
    context += i 

# issues: 
## 1. sometimes a sentence (e.g. with a comma) actually gives two answers

# save 
with open(f'data/text-davinci-003_phillips2017_mary_branches.json', 'w') as f:
  json.dump(list_sentence_context, f)