import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import itertools
from sklearn.metrics.pairwise import cosine_distances
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
from helper_functions import *

# setup
num_per_iteration=20

# load data
df = pd.read_csv('../data/data_cleaned/gpt4.csv')

# sort values
df = df.sort_values(['condition', 'id', 'iteration', 'num'])

### 1. similarity against first generation ###
### done for each (condition, id) pair ###
id_list = df['id'].unique()
condition_list = df['condition'].unique()
combinations = list(itertools.product(id_list, condition_list))

for id, condition in combinations:
    # get responses
    responses = df[(df['id'] == id) & (df['condition'] == condition)]['response'].tolist()
    # encode sentences (see helper_functions.py)
    encodings = encode_responses(tokenizer, responses)
    # embed responses (see helper_functions.py)
    embeddings = embed_responses(model, encodings)
    # get cosine distances in the correct format 
    num_rows = embeddings.shape[0]
    num_iterations = int(num_rows / num_per_iteration)
    reshaped_embeddings = embeddings.reshape(num_iterations, num_per_iteration, 384)
    mean_cosine_distances = []
    nums = []
    froms = []
    tos = []
    cosine_distances_list = []
    for i in range(num_per_iteration):
        col_elements = reshaped_embeddings[:, i, :]
        distances = cosine_distances(col_elements)
        for from_idx in range(len(distances)):
            for to_idx in range(len(distances)):
                nums.append(i+1) # +1 just for prettier plot
                froms.append(from_idx)
                tos.append(to_idx)
                cosine_distances_list.append(distances[from_idx, to_idx])
    # Create the DataFrame
    df_plot = pd.DataFrame({
        'num': nums,
        'from': froms,
        'to': tos,
        'cosine_distance': cosine_distances_list
    })
    # Plot 
    fig, ax = plt.subplots()
    sns.boxplot(x='num', y='cosine_distance', data=df_plot)
    plt.title(f'{id} {condition}')
    plt.savefig(f'../fig/gpt4/first_response/{id}_{condition}.png')

# then we take responses out 
responses = df['response'].tolist()

# encode sentences (see helper_functions.py)
encodings = encode_responses(tokenizer, responses)

# embed responses (see helper_functions.py)
embeddings = embed_responses(model, encodings)

### 1. how similar is each first, second, n element to each other ###

num_per_iteration = 20
num_rows = embeddings.shape[0]
num_iterations = int(num_rows / num_per_iteration)
reshaped_embeddings = embeddings.reshape(num_iterations, num_per_iteration, 384)
mean_cosine_distances = []
nums = []
froms = []
tos = []
cosine_distances_list = []
for i in range(num_per_iteration):
    col_elements = reshaped_embeddings[:, i, :]
    distances = cosine_distances(col_elements)
    
    for from_idx in range(len(distances)):
        for to_idx in range(len(distances)):
            nums.append(i+1) # +1 just for prettier plot
            froms.append(from_idx)
            tos.append(to_idx)
            cosine_distances_list.append(distances[from_idx, to_idx])

# Create the DataFrame
df = pd.DataFrame({
    'num': nums,
    'from': froms,
    'to': tos,
    'cosine_distance': cosine_distances_list
})

import seaborn as sns 
sns.boxplot(x='num', y='cosine_distance', data=df)
