'''
VMP 2023-05-23:
works with the nlp conda environment now on GPU. 
'''
import torch
import torch.nn.functional as F
import numpy as np
import scipy 
import pandas as pd 
import re 
from tqdm import tqdm
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
print(torch.__version__) 
print(torch.version.cuda)
# setup 
document_path = 'data'
figure_path = 'fig'
document_name = 'ted_chiang_hell_absence'

# device setup
device=torch.device('cuda:0')
enc = GPT2Tokenizer.from_pretrained('gpt2') 
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

context = enc.encode("I am")
context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    for _ in range(1000):
        logits, _ = model(context, past=None)
    
# key function 
def p_next_token(sentence, nmax=50257):
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

# load document
with open(f'{document_path}/{document_name}.txt', 'r') as f:
    document = f.read().replace("\n", " ")
    document = document.replace("’", "'")
 
reg = r"(\s*\w+|\s*\b'\w*|\s*[.,?!:;])"
sentence_list=re.findall(reg, document)

# get next word probability for each word in sentence
pdict={}
running_sentence = ""
for i in tqdm(range(len(sentence_list)-1)): 
    running_sentence += sentence_list[i]
    next_word = sentence_list[i+1]
    pdist=p_next_token(running_sentence)
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
        
# to dataframe
df = pd.DataFrame.from_dict(pdict, 
                            orient='index', 
                            columns=['word', 'match', 'p_token', 'entropy'])
df['complete'] = df['word'] == df['match']

# initial plotting 
df[df['complete']==False]



import matplotlib.pyplot as plt

# Create a new figure and an axes
fig, ax1 = plt.subplots(figsize=(40, 6), dpi=300)

# Plot p_token
ax1.plot(df.index, df['p_token'], color='tab:blue')
ax1.set_xlabel('Token')
ax1.set_ylabel('p_token', color='tab:blue')
ax1.tick_params('y', colors='tab:blue')
ax1.scatter(df.index, df['p_token'], color='tab:blue', s=15)

for i, highlight in enumerate(df['complete']):
    if not highlight:
        ax1.axvspan(i-0.5, i+0.5, color='tab:red', alpha=0.3)

# rotate x axis labels
plt.xticks(df.index, df['word'].tolist(), rotation=90)

# Create a second y-axis that shares the same x-axis as ax1
ax2 = ax1.twinx()

# Plot entropy
ax2.plot(df.index, df['entropy'], color='tab:orange')
ax2.set_ylabel('entropy', color='tab:orange')
ax2.tick_params('y', colors='tab:orange')
ax2.scatter(df.index, df['entropy'], color='tab:orange', s=15)

# Show the plot
plt.tight_layout()
plt.savefig(f'{figure_path}/{document_name}_p_token_entropy.png', dpi=300)


##### reverse code ######
df['-p_token'] = -df['p_token']

# Create a new figure and an axes
fig, ax1 = plt.subplots(figsize=(40, 6), dpi=300)

# Plot p_token
ax1.plot(df.index, df['-p_token'], color='tab:blue')
ax1.set_xlabel('Token')
ax1.set_ylabel('-p_token', color='tab:blue')
ax1.tick_params('y', colors='tab:blue')
ax1.scatter(df.index, df['-p_token'], color='tab:blue', s=15)

for i, highlight in enumerate(df['complete']):
    if not highlight:
        ax1.axvspan(i-0.5, i+0.5, color='tab:red', alpha=0.3)

# rotate x axis labels
plt.xticks(df.index, df['word'].tolist(), rotation=90)

# Create a second y-axis that shares the same x-axis as ax1
ax2 = ax1.twinx()

# Plot entropy
ax2.plot(df.index, df['entropy'], color='tab:orange')
ax2.set_ylabel('entropy', color='tab:orange')
ax2.tick_params('y', colors='tab:orange')
ax2.scatter(df.index, df['entropy'], color='tab:orange', s=15)

# Show the plot
plt.tight_layout()
plt.savefig(f'{figure_path}/{document_name}_-p_token_entropy.png', dpi=300)
