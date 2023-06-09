'''
Generates the completions used in the submission. 
'''
import openai 
import os 
from dotenv import load_dotenv
import json 
from tqdm import tqdm 
import re 
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

inpath='../data/data_input/phillips2017.json'
outpath='../data/data_output/phillips_gpt3/'
condition='should' # could
num_generations=100
num_times=10

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
prompts = ['One thing that Heinz could do is to',
           'One thing that Josh could do is to',
           'One thing that Brian could do is to',
           'One thing that Liz could do is to',
           'One thing that Mary could do is to',
           'One thing that Brad could do is to']
if condition == 'should':
    prompts = [re.sub('could', 'should', s) for s in prompts]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_completion(context, temperature, num_generations):
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=context,
        max_tokens=200, # one sentence should not run longer than this 
        stop=['.', '?', '!', '\n\n'], # stop at first period (or similar)
        temperature=temperature,
        n=num_generations
        )
    return completion 

def generate_completions_batched(vignettes, prompts, num_generations, temperature):
    contexts = [background + prompt for background, prompt in zip(vignettes, prompts)]
    generation_dict = {}
    for num, context in tqdm(enumerate(contexts)): 
        completion = create_completion(context, temperature, num_generations)
        
        generation_list = [completion['choices'][num]['text'] for num in range(num_generations)]
        
        # ensuring responsese at least length 2
        # i.e., some responses start with e.g. "1. " or newline. 
        # regenerate these responses. 
        for i, ele in enumerate(generation_list): 
            if len(ele.split()) < 2: 
                print('less than 2 words')
                length=1
                while length < 2:
                    print('enter while') 
                    completion_fix = create_completion(context, temperature, 1)
                    text = completion_fix['choices'][0]['text']
                    generation_list[i]=text
                    length = len(text.split())
                
        generation_dict['context_' + str(num)] = {}
        generation_dict['context_' + str(num)]['vignette'] = vignettes[num]
        generation_dict['context_' + str(num)]['prompt'] = prompts[num]
        generation_dict['context_' + str(num)]['generation'] = generation_list
    return generation_dict

temperature = 1.0 # [0.5, 1.0]
for i in range(num_times):
    #for temperature in temperature_grid: 
    generation_dict = generate_completions_batched(vignettes, prompts, num_generations, temperature)
    path_string = f'phillips2017_text-davinci-003_n{num_generations}_m{i}_temp{temperature}_{condition}_fix.json'
    out_string = os.path.join(outpath, path_string)
    with open(out_string, 'w') as fp:
        json.dump(generation_dict, fp)