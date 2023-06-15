from sklearn.decomposition import PCA
import json 
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd 
import numpy as np 
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
import seaborn as sns 
import matplotlib.pyplot as plt

# load data
n=10
with open(f'../gpt/data/text-davinci-003_phillips2017_n{n}_could_should.json') as user_file:
  phillips2017 = json.load(user_file)

# this is actually a pretty bad setup
contexts = list(phillips2017.keys())
conditions = ['could', 'should']

sentences = [x.values() for x in phillips2017.values()]
flattened = [item for sublist in sentences for value in sublist for item in value]
cleaned = [s.strip() for s in flattened]

# load computed probabilities and entropies
with open(f'data_output/text-davinci-003_phillips2017_n{n}_prob_ent.json') as user_file:
  probability_entropy = json.load(user_file)

# create dataframe with values 
word_list = []
match_list = []
probability_list = []
entropy_list = []
index_list = []

for outer_key in sorted(probability_entropy.keys()):
    inner_values = probability_entropy[outer_key].values()
    inner_word = [value[0] for value in inner_values]
    inner_match = [value[1] for value in probability_entropy[outer_key].values()]
    inner_probability = [value[2] for value in probability_entropy[outer_key].values()]
    inner_entropy = [value[3] for value in probability_entropy[outer_key].values()]

    word_list.append(inner_word)
    match_list.append(inner_match)
    probability_list.append(inner_probability)
    entropy_list.append(inner_entropy)
    index_list.append([outer_key for _ in range(len(inner_values))])

df_values = pd.DataFrame(
    {'index': [int(item) for sublist in index_list for item in sublist],
     'word': [item for sublist in word_list for item in sublist],
     'match': [item for sublist in match_list for item in sublist],
     'probability': [item for sublist in probability_list for item in sublist],
     'entropy': [item for sublist in entropy_list for item in sublist]
    })

# create dataframe with contexts 
num_generation=10
num_contexts=6
context_lst = [item for item in contexts for _ in range(num_generation)]
conditions_lst = [item for item in conditions for _ in range(num_generation)]
conditions_lst = conditions_lst * num_contexts

df_context = pd.DataFrame({
    'context': context_lst,
    'condition': conditions_lst
})
df_context['index'] = df_context.index
df_context['context'] = df_context['context'].str.split().str[0]

# master dataframe
df_master = pd.merge(df_values, df_context, on='index')

# quick calculations
## really looks like it is just exactly the same for the mean
## perhaps this has to be the case (i.e. based on the heat is
## it chooses words that are equally likely?) 
df_agg = df_master.groupby(['context', 'condition']).agg(
    {'probability': ['mean', 'std', 'min', 'max', 'median'], 
     'entropy': ['mean', 'std', 'min', 'max', 'median']})

# Questions
## mean does not give probability of sentence 
## we would need to multiply and normalize in some way
## also these are already logprobs 
## not quite sure what makes sense here?





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