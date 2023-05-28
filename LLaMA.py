from llama_cpp import Llama
import re
#import numpy as np 
#import torch.nn.functional as F
#import torch 

# how do we get the GPU working here?

# should import 4-bit instead 
llm = Llama(model_path="llama.cpp/models/7B/ggml-model-q4_0.bin",
            logits_all=True, # key to getting probs
            verbose=True,
            n_gpu_layers=4) 

# we just cannot get logprobs 
n_vocab = llm.n_vocab()
sentence="I am"
completion = llm.create_completion(
    prompt=sentence,
    max_tokens=100,
    temperature=0.6,
    logprobs=n_vocab
)

def next_word(sentence, nmax=50257):
    completion=llm.create_completion(sentence,
                            max_tokens=1,
                            temperature=0.6,
                            logprobs=n_vocab)
    top_logprobs=completion['choices'][0]['logprobs']['top_logprobs'][0]
    softmax_probs=F.softmax(torch.tensor(list(top_logprobs.values())), dim=-1)
    top_logprobs=dict(zip(list(top_logprobs.keys()), softmax_probs.tolist()))
    #should already be sorted but just for good measure
    top_logprobs=dict(sorted(top_logprobs.items(), key=lambda item: item[1], reverse=True))
    #divisor=np.sum([y for _, y in top_logprobs.items()])
    return top_logprobs #{i[0]: i[1]/divisor for i in top_logprobs}

sentence='I am'
completion=llm.create_completion(sentence,
                        max_tokens=1,
                        temperature=0.6,
                        logprobs=n_vocab)
top_logprobs=completion['choices'][0]['logprobs']['top_logprobs'][0]
softmax_probs=F.softmax(torch.tensor(list(top_logprobs.values())), dim=-1)
top_logprobs=dict(zip(list(top_logprobs.keys()), softmax_probs.tolist()))

# test on sentence  
original_sentence = 'Anyone else only feel happy when they are asleep and the minute you wake up you remember how much you hate your life and instantly feel like your soul is filled with cement?'
# split sentence and keep leading spaces and separate punctuation (is this what we want?)
sentence_list = re.findall(r'(\s*\w+|\s*\.)', original_sentence)
# get next word probability for each word in sentence
pdict={}
running_sentence = ""
for i in range(len(sentence_list)-1): 
    running_sentence += sentence_list[i]
    pdist=next_word(running_sentence, nmax=n_vocab)
    pnext=pdist.get(sentence_list[i+1])
    pdict[sentence_list[i+1]]=pnext

# quick plot
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
plt.plot(pdict.values())
plt.xticks(range(len(pdict)), pdict.keys(), rotation=90)
