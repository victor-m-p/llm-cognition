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
#n=10
with open(f'../gpt/data/text-davinci-003_phillips2017_josh_branches.json') as user_file:
  phillips2017 = json.load(user_file)

# this is actually a pretty bad setup
#contexts = list(phillips2017.keys())
#conditions = ['could', 'should']

#sentences = [x.values() for x in phillips2017.values()]
#flattened = [item for sublist in sentences for value in sublist for item in value]
#cleaned = [s.strip() for s in flattened]

# load computed probabilities and entropies
with open(f'data_output/text-davinci-003_phillips2017_josh_branches.json') as user_file:
  probability_entropy = json.load(user_file)

# create dataframe with values 
word_list = []
match_list = []
probability_list = []
entropy_list = []
sentence_index_list = []
token_index_list = []

probability_entropy.keys()

for outer_key in probability_entropy.keys():
    inner_values = probability_entropy[outer_key].values()
    inner_word = [value[0] for value in inner_values]
    inner_match = [value[1] for value in probability_entropy[outer_key].values()]
    inner_probability = [value[2] for value in probability_entropy[outer_key].values()]
    inner_entropy = [value[3] for value in probability_entropy[outer_key].values()]

    word_list.append(inner_word)
    match_list.append(inner_match)
    probability_list.append(inner_probability)
    entropy_list.append(inner_entropy)
    sentence_index_list.append([outer_key for _ in range(len(inner_values))])
    token_index_list.append([i for i in range(len(inner_values))])

df_values = pd.DataFrame(
    {'sentence_index': [int(item) for sublist in sentence_index_list for item in sublist],
     'token_index': [int(item) for sublist in token_index_list for item in sublist],
     'word': [item for sublist in word_list for item in sublist],
     'match': [item for sublist in match_list for item in sublist],
     'probability': [item for sublist in probability_list for item in sublist],
     'entropy': [item for sublist in entropy_list for item in sublist]
    })

# create dataframe with contexts 
wordsplit_index = [[i for j, completion in enumerate(sublist)] for i, sublist in enumerate(phillips2017)]
wordsplit_index = [item for sublist in wordsplit_index for item in sublist]
sentence_index = [i for i in range(len(wordsplit_index))]

df_context = pd.DataFrame({
    'word_split': wordsplit_index,
    'sentence_index': sentence_index
})

#df_context['sentence_index'] = df_context.index
#df_context['context'] = df_context['context'].str.split().str[0]

# master dataframe
df_master = pd.merge(df_values, df_context, on='sentence_index')
df_master = df_master.sort_values(by=['sentence_index', 'token_index'])

# quick calculations
## here the actual sentence chosen is very improbable 
## apparently; compared with the mean of the other 
## variations that we have. 
df_agg = df_master.groupby(['word_split']).agg(
    {'probability': ['mean', 'std', 'min', 'max', 'median'], 
     'entropy': ['mean', 'std', 'min', 'max', 'median']})

## plot over time 
# Create a new figure and an axes
#fig, ax1 = plt.subplots(figsize=(40, 6), dpi=300)

# Plot p_token
df_master['word_split'] = df_master['word_split']
sns.lineplot(data=df_master,
             x='token_index',
             y='probability',
             hue='word_split',
             estimator=None,
             units='sentence_index',
             palette='viridis',
             alpha=0.3)

sns.lineplot(data=df_master,
             x='token_index',
             y='entropy',
             hue='word_split',
             estimator=None,
             units='sentence_index',
             palette='viridis',
             alpha=0.3)