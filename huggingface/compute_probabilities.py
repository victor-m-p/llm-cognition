'''
python gpt2.py -i data_input/ -o data_output/
getting logprobs and entropy of sequence. 
not used in submitted paper. 
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_folder', required = True, type = str, help = 'path to input .txt files')
    ap.add_argument('-o', '--output_folder', required = True, type = str, help = 'path to output .json files')
    args = vars(ap.parse_args())
    main(
        input_folder = args['input_folder'],
        output_folder = args['output_folder']
    )