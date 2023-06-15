'''
python gpt2.py -i data_input/ -o data_output/
getting logprobs and entropy of sequence. 
'''

import re 
from tqdm import tqdm 
import numpy as np 
import scipy 
import json
import torch 
from torch.functional import F
import os 
import argparse 
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = 'decapoda-research/llama-7b-hf' # what we want to try 
checkpoint = 'gpt2'

def p_next_token(sentence, model, tokenizer, device, nmax=50257):
    inputs = tokenizer(sentence, return_tensors="pt").to(device) # do we have to do this for every one?
    with torch.no_grad(): # does this reduce time?
        outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    logits=F.softmax(logits[:, -1, :], dim=-1)
    pw=zip([tokenizer.decode([i]) for i in range(logits[0].size()[0])], logits[0].tolist())
    pw=sorted(pw, key = lambda x: -x[1])
    pw=pw[0:nmax]
    divisor=np.sum([y for _, y in pw])
    return {i[0]: i[1]/divisor for i in pw}

# get next word probability for each word in sentence
def document_probabilities(context, sentence_list, model, tokenizer, device):
    pdict={}
    for i in tqdm(range(len(sentence_list)-1)): 
        next_word = sentence_list[i]
        pdist=p_next_token(context, model, tokenizer, device)
        entropy=scipy.stats.entropy(list(pdist.values()))
        pnext=pdist.get(next_word)	
        # if we match the word fine
        if pnext is not None:
            pdict[i]=[next_word, next_word, pnext, entropy]
        # else find longest subword that matches token
        else: 
            sub_word = next_word
            while pnext is None:
                sub_word = sub_word[:-1]
                pnext=pdist.get(sub_word)
            pdict[i]=[next_word, sub_word, pnext, entropy]
        # add next word to context 
        context += sentence_list[i]
    return pdict

def clean_split_string(string):
    reg = r"(\s*\w+|\s*\b'\w*|\s*[.,?!:;])"
    sentence_list=re.findall(reg, string)
    return sentence_list

# setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}")
model = AutoModelForCausalLM.from_pretrained(f"{checkpoint}")
model = model.to(device)

# load document 
num_per_group=10
with open(f'../gpt/data/text-davinci-003_phillips2017_n{num_per_group}_could_should.json') as user_file:
  phillips2017 = json.load(user_file)

# wrangling (not the best setup currently)
contexts = list(phillips2017.keys())
contexts_lst = [item for item in contexts for _ in range(num_per_group)]

# clean inputs
answers_raw = [x.values() for x in phillips2017.values()]
answers_flat = [item for sublist in answers_raw for value in sublist for item in value]
answers_clean = [s.strip() for s in answers_flat]
answers_split = [clean_split_string(x) for x in answers_clean]

overall_dict = {}
for i, zip_ele in enumerate(zip(contexts_lst, answers_split)):
    context, answer = zip_ele
    pdict = document_probabilities(context, answer, model, tokenizer, device)
    overall_dict[i] = pdict
    
# save this object
with open(f'data_output/text-davinci-003_phillips2017_n{num_per_group}_prob_ent.json', 'w') as fp:
    json.dump(overall_dict, fp)