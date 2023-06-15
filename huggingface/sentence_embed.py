'''
https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
testing sBERT embeddings. 
probably legacy at this point
'''

import json 
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd 
import numpy as np 
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 

model = AutoModel.from_pretrained(checkpoint) # which model?
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open('data_input/phillips2017.json') as user_file:
  phillips2017 = json.load(user_file)
phillips2017C1 = phillips2017['context_6']

sentences = []
idxs = []
for key, val in phillips2017C1.items():
    if isinstance(val, list):
        sentences += val
        idxs += [key for _, _ in enumerate(val)]
    else: 
        sentences.append(val)
        idxs.append(key)
        
df_idx = pd.DataFrame(idxs)
df_idx.columns = ['type']
df_idx['idx'] = df_idx.index

encoded_input = tokenizer(sentences, 
                          padding=True, 
                          truncation=True, 
                          #max_length=128, -- what they train with
                          return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Compute pairwise cosine distances
distances = cosine_distances(sentence_embeddings) # cosine dist. between all pairs

# Convert the resulting numpy array to a pandas DataFrame
df = pd.DataFrame(distances)

# Get the upper triangle of the DataFrame
# We use np.triu_indices on the DataFrame shape and exclude the diagonal with k=1
upper_triangle = list(zip(*np.triu_indices(df.shape[0], k=1)))

# Create a new DataFrame in the long format
df_long = pd.DataFrame(upper_triangle, columns=['sentence_x', 'sentence_y'])

# Populate the cosine_distance column
df_long['cosine_distance'] = [df.loc[x, y] for x, y in upper_triangle]

# joining this; 
def merge_dfs(df_main, df_aux, main_col, aux_col, new_col_name):
    df_main = pd.merge(df_main, df_aux, left_on=main_col, right_on=aux_col, how='left')
    df_main.rename(columns={'type': new_col_name}, inplace=True)
    df_main.drop(aux_col, axis=1, inplace=True)
    return df_main

df_main = merge_dfs(df_long, df_idx, 'sentence_x', 'idx', 'type_x')
df_main = merge_dfs(df_main, df_idx, 'sentence_y', 'idx', 'type_y')

# try to plot something
# wow; does not track this well at all 
import matplotlib.pyplot as plt
import seaborn as sns 

df_vignette = df_main[df_main['type_x'] == 'vignette']
sns.boxplot(data=df_vignette,
            x = 'type_y',
            y = 'cosine_distance',
            palette='Blues')

df_ordinary = df_main[df_main['type_x'] == 'ordinary']
sns.boxplot(data=df_ordinary,
            x = 'type_y',
            y = 'cosine_distance',
            palette='Blues')