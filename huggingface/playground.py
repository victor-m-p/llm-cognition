'''
python gpt2.py -i data_input/ -o data_output/
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

# imports and setup for huggingface
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}")
model = AutoModelForCausalLM.from_pretrained(f"{checkpoint}")
model = model.to(device)

# checking things: 
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(device)
outputs = model(**inputs, 
                labels=inputs["input_ids"], # gives us loss
                output_hidden_states=True, # gives us hidden states
                output_attentions=True # gives us attentions
                ) 

# getting logits
outputs.logits.shape # logits

# getting loss (do we want this; can we maybe make sure not to use?)
outputs.loss.shape # loss

# hidden states
outputs.hidden_states[12].shape # (1, 6, 768) x 13 layers 

# last hidden state 
outputs.last_hidden_state.shape # nope

# past key values
outputs.past_key_values[0][1].shape #(1, 12, 6, 64)
outputs.past_key_values[11][0]
# 12 x 2 x (1, 12, 6, 64)

# do we have attentions?
outputs.attentions[11].shape # (1, 12, 6, 6)

# checking model
model.eval() # 
model.num_parameters() # 124,439,808


''' Questions: 
https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutput
1. what does "labels" do in model(): give us loss (which we probably do not need)
2. how is it different from: labels = torch.tensor([1]).unsqueeze(0)
3. what do we get from returning tensors? 
4. whicih objects do we have in "outputs"?
'''


def p_next_token(sentence, model, tokenizer, device, nmax=50257):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    logits=F.softmax(logits[:, -1, :], dim=-1)
    pw=zip([tokenizer.decode([i]) for i in range(logits[0].size()[0])], logits[0].tolist())
    pw=sorted(pw, key = lambda x: -x[1])
    pw=pw[0:nmax]
    divisor=np.sum([y for _, y in pw])
    return {i[0]: i[1]/divisor for i in pw}

# get next word probability for each word in sentence
def document_probabilities(sentence_list, model, tokenizer, device):
    pdict={}
    running_sentence = ""
    for i in tqdm(range(len(sentence_list)-1)): 
        running_sentence += sentence_list[i]
        next_word = sentence_list[i+1]
        pdist=p_next_token(running_sentence, model, tokenizer, device)
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
    return pdict

# load document
def load_file(input_path, document_name):
    input_file = os.path.join(input_path, document_name)
    with open(input_file, 'r') as f:
        document = f.read().replace("\n", " ")
        document = document.replace("’", "'")
    reg = r"(\s*\w+|\s*\b'\w*|\s*[.,?!:;])"
    sentence_list=re.findall(reg, document)
    return sentence_list

# run main function
def main(input_folder, output_folder): 
    # imports and setup for huggingface
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(f"{checkpoint}")
    model = model.to(device)

    # run the pipeline over all documents in input path 
    documents = os.listdir(input_folder)
    for document_name in documents:
        sentence_list = load_file(input_folder, document_name)
        pdict = document_probabilities(sentence_list, model, tokenizer, device)
        document_name = document_name.replace('.txt', 'json')
        output_file = os.path.join(output_folder, document_name)
        with open(output_file, 'w') as fp:
            json.dump(pdict, fp)