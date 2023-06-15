'''
efficiently generate text from a given prompt. 
with gpt2 this does not work well unfortunately. 
the model is just not good enough to generate 
something coherent. at least not for a prompt of
this length. 
'''

import json 
from transformers import pipeline, set_seed
checkpoint = 'decapoda-research/llama-7b-hf' # what we want to try 
checkpoint = 'gpt2'
set_seed(42)
import xformers # just needs to be somewhere for pipreqs to understand that it should generate it in the requirements.txt
generator = pipeline('text-generation', model=checkpoint)
prompt = 'Sam is 30 years old, and has a close friend, Harry, who is struggling in life. Harry has problems with addiction, and often feels lonely and unappreciated. Harry wants to get better, but is not doing as well as he could be given his innate abilities. Sam sets a goal of helping Harry become a happier and more thriving person. The first thing Sam could do is to'
out = generator(prompt, max_length=500, num_return_sequences=5)

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

num_generations = 2
max_tokens = 100 # sentences typically < 30 tokens
results = {}
for pairs in all_contexts: 
    condition, prompt = pairs 
    generation_list = []
    for i in range(num_generations): 
        out = generator(prompt, max_length=max_tokens, num_return_sequences=5)

        completion = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.6
            #logprobs=n_vocab
        )
        text = completion['choices'][0]['text']
        text = get_first_sentence(text)
        generation_list.append(text)
    results[prompt] = {}
    results[prompt][condition] = generation_list

out = generator(prompt, max_length=500, num_return_sequences=5)
