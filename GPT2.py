'''
VMP 2023-05-23:
working through some of Simon's code to understand and clean.
'''
import torch
import torch.nn.functional as F
import numpy as np
import sys
import re 
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

device=torch.device("cpu")

if hasattr(torch.backends, 'mps'):
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # torch.device('mps') #

enc = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

def next_word(sentence, nmax=50257):
    # encode sentence
	context_tokens = enc.encode(sentence)
	# limit context to 1023 tokens (why?)
	if (len(context_tokens) > 1023): 
		context_tokens=context_tokens[-1023:]
		print(sentence)
		print(context_tokens)
	# context (not quite sure what this is)
	context = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
	# compute logits for next word
	with torch.no_grad():
		logits, _ = model(context, past=None)
	logits=F.softmax(logits[:, -1, :], dim=-1)
	# get the decoded words and logits
	pw=zip([enc.decode([i]) for i in range(logits[0].size()[0])], logits[0].tolist())
	pw=sorted(pw, key = lambda x: -x[1])
	# get only 50257 elements (already seems to be the case)
	pw=pw[0:nmax] 
	# get divisor for normalization 
	divisor=np.sum([y for _, y in pw])
	# return dictionary with key = word, value = probability
	return {i[0]: i[1]/divisor for i in pw}

# test on sentence  
original_sentence = 'Anyone else only feel happy when they are asleep and the minute you wake up you remember how much you hate your life and instantly feel like your soul is filled with cement?'
# split sentence and keep leading spaces and separate punctuation (is this what we want?)
sentence_list = re.findall(r'(\s*\w+|\s*\.)', original_sentence)
# get next word probability for each word in sentence
pdict={}
running_sentence = ""
for i in range(len(sentence_list)-1): 
    running_sentence += sentence_list[i]
    print(running_sentence)
    pdist=next_word(running_sentence)
    pnext=pdist.get(sentence_list[i+1])
    pdict[sentence_list[i+1]]=pnext

# quick plot
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
plt.plot(pdict.values())
plt.xticks(range(len(pdict)), pdict.keys(), rotation=90)


# encode sentence
sentence = "I love to arrange things" 
context_tokens = enc.encode(sentence)
# limit context to 1023 tokens (why?)
if (len(context_tokens) > 1023): 
	context_tokens=context_tokens[-1023:]
	print(sentence)
	print(context_tokens)
# context (not quite sure what this is)
context = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
# compute logits for next word
with torch.no_grad():
	logits, _ = model(context, past=None)
logits.shape
logits=F.softmax(logits[:, -1, :], dim=-1)
logits.shape
# get the decoded words and logits
pw=zip([enc.decode([i]) for i in range(logits[0].size()[0])], logits[0].tolist())
pw=sorted(pw, key = lambda x: -x[1])
words = [x for x, y in pw]
words = [x for x in words if x == ' delicately']

## not clear to me what this is for ## 
def how_likely_terminal(sentence):
	ans=next_word(sentence, 10) # this is set of predictions
	counter=0
	for i in ans: ## for each prediction, let's see what is next...
		running_word=i[0]
		running_prob=i[1]
		if not(running_word[0].isalpha()):
			counter += running_prob
	return counter

def next_word_terminal(sentence, nmax=1000):
	ans=next_word(sentence, nmax)
	final=[]
	for i in ans: ## for each prediction, let's see what is next...
		running_word=i[0]
		running_prob=i[1]
		if how_likely_terminal(sentence+running_word) > 0.5:
			final.append((running_word, running_prob))
	count=0

	for i in final:
		count += i[1]
	final=[[i[0], i[1]/count] for i in final]
	return final

while(True):
	for line in sys.stdin:
		print(next_word(line.rstrip()), flush=True)