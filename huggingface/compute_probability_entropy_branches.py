'''
python gpt2.py -i data_input/ -o data_output/
getting logprobs and entropy of sequence. 
not currently used. 
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
with open(f'../gpt/data/text-davinci-003_phillips2017_josh_branches.json') as user_file:
  phillips2017 = json.load(user_file)

# wrangling (not the best setup currently)
index = [[i for j, completion in enumerate(sublist)] for i, sublist in enumerate(phillips2017)]
index = [item for sublist in index for item in sublist]

sentences = [[completion for completion in sublist] for sublist in phillips2017]
sentences = [item for sublist in sentences for item in sublist]
sentences_split = [clean_split_string(x) for x in sentences]

context = "Josh is on the way to the airport to catch a flight for a hunting safari in Africa. He leaves with plenty of time to make it there, but his car breaks down on the highway. Now Josh is sitting in his car near a busy intersection, and knows he needs to get to airport soon if he is going to catch his flight."
contexts = [context for i in range(len(sentences_split))]

overall_dict = {}
for i, zip_ele in enumerate(zip(contexts, sentences_split)):
    context, answer = zip_ele
    pdict = document_probabilities(context, answer, model, tokenizer, device)
    overall_dict[i] = pdict

# save this object
with open(f'data_output/text-davinci-003_phillips2017_josh_branches.json', 'w') as fp:
    json.dump(overall_dict, fp)