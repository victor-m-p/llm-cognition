import openai 
import pandas as pd
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

outpath='../data/data_output/gpt4_eval/'
model='gpt-4'
temperature=0.8
frequency=0.0
presence=0.0

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

df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def prepare_prompt(group):
    vignette = group['vignette'].iloc[0]
    id = group['id'].iloc[0]
    prompt = f'Here are six suggestions for what {id} could do. Please, rank them from best to worst. Your response should be a list of numbers, in the order from best to worst, where the best is first and the worst is last.\n'
    options = group['response_option'].tolist()
    options = [s.strip() for s in options]
    options = [f"{i+1}. {item}\n" for i, item in enumerate(options)]
    full_prompt = vignette + prompt + ''.join(options)
    condition = group['condition'].iloc[0]
    if condition == 'should': 
        full_prompt = re.sub('could', 'should', full_prompt)
    return full_prompt 

df = df.sort_values(by=['condition', 'id', 'iteration', 'shuffled'])
all_prompts = df.groupby(['condition', 'id', 'iteration']).apply(prepare_prompt)

# 1. makeshift shelter (4)-3
# 2. higher ground (5)-4
# 3. fire warmth (3)-2
# 4. fresh water (2)-6
# 5. ration remaining food (1)-5
# 6. electronic devices (6)-1
#all_prompts[0]
#completion=create_completion(all_prompts[0], model, 1,
#                             temperature, frequency, presence)
#completion['choices'][0]['message']['content']

# test that we get that ordering "forward":
#background = "Brad and some friends are hiking through the mountains in the Canadian wilderness. A couple of days into their hike, Brad realizes that they are lost. He knows that a rescue crew could arrive before long, but it is extremely cold and they don't have much food or water left"
#prompt = 'Please come up with 6 things that Brad could do in this situation. Number them like this: 1. one thing that they could do is ... and so on. Always begin each number with the phrase "one thing that they could do is"'
#both = background + prompt
#test = create_completion(both, model, 1,
#                            temperature, frequency, presence)
#test['choices'][0]['message']['content']
#x = create_completion(all_prompts, model, 5,
#                      temperature, frequency, presence)
#x['choices'][4]['message']['content']

generation_list_outer = []
for prompt in all_prompts:
    completion = create_completion(prompt, model, 1,
                                   temperature, frequency, presence)
    generation_list_inner = completion['choices'][0]['message']['content']
    generation_list_outer.append(generation_list_inner)

# check that this is not just complete bogus. 
prompts=all_prompts.reset_index(name='prompt')
prompts.to_csv('../data/data_output/gpt4_eval/prompts.csv', index=False)

'''
# average distance
# ikke vanvittigt staerk correlation. 
import numpy as np 
distances = {}

columns = ['num', 'shuffled', 'ranking']

for i, col1 in enumerate(columns):
    for j, col2 in enumerate(columns):
        if i >= j:
            continue
        distance = np.mean(np.abs(test[col1] - test[col2]))
        distances[f"{col1} - {col2}"] = distance

# Print out the average distances between each pair of columns
for pair, distance in distances.items():
    print(f"Average distance between {pair}: {distance}")

# 1.944 is completely random so 1.4-something is a weak signal. 
# seems like it is using both the presentation order and the content.
# so this is basically random which is weird. 
'''