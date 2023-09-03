'''
VMP 2023-08-29
Figure out what happens within generations and across iterations
(but not across conditions in this script).
1. similarity against first generation
3. heatmat of similarity within generations between num
'''

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
# 1. gpt4.csv and n_per_iteration=20
# 2. gpt4_subset.csv and n_per_iteration=18 (removeing first and last option)
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
num_per_iteration=6

# sort values
df = df.sort_values(['condition', 'id', 'iteration', 'num'])

# all combinations of (id, condition)
id_list = df['id'].unique()
condition_list = df['condition'].unique()
combinations = list(itertools.product(id_list, condition_list))

### 1. similarity against first generation ###
for id, condition in combinations:
    # get responses
    responses = df[(df['id'] == id) & (df['condition'] == condition)]['response_option'].tolist()
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
    plt.savefig(f'../fig/gpt4/same_num_across_iter/{id}_{condition}.png')

### 2. 
iterations = []
froms = []
tos = []
cosine_distances_list = []
for id, condition in combinations:
    # get responses
    responses = df[(df['id'] == id) & (df['condition'] == condition)]['response_option'].tolist()
    # encode sentences (see helper_functions.py)
    encodings = encode_responses(tokenizer, responses)
    # embed responses (see helper_functions.py)
    embeddings = embed_responses(model, encodings)
    # get cosine distances in the correct format 
    num_rows = embeddings.shape[0]
    num_iterations = int(num_rows / num_per_iteration)
    reshaped_embeddings = embeddings.reshape(num_iterations, num_per_iteration, 384)
        
    # Loop through the 41 iterations
    for iter_idx in range(num_iterations):
        # Loop through the 20 columns to compute the pairwise cosine distance
        for col_idx in range(num_per_iteration):
            # The element we will compare from (always the first in each 20-element set)
            from_element = reshaped_embeddings[iter_idx, 0, :].reshape(1, -1)
            
            # The element we will compare to
            to_element = reshaped_embeddings[iter_idx, col_idx, :].reshape(1, -1)
            
            # Compute cosine distance
            distance = cosine_distances(from_element, to_element)
            
            # Store in lists
            iterations.append(iter_idx)
            froms.append(0)  # Always comparing from the first element in each set
            tos.append(col_idx)
            cosine_distances_list.append(distance[0][0])

    # Create the DataFrame
    df_ = pd.DataFrame({
        'cosine_distance': cosine_distances_list,
        'iteration': iterations,
        'from': froms,
        'to': tos
    })

    fig, ax = plt.subplots()
    sns.boxplot(x='to', y='cosine_distance', data=df_)
    plt.xlabel('iteration')
    plt.savefig(f'../fig/gpt4/dist_first_num/{id}_{condition}.png')

### 3. heatmat of similarity within generations between num ###
for id, condition in combinations:
    # get responses
    responses = df[(df['id'] == id) & (df['condition'] == condition)]['response_option'].tolist()
    # encode sentences (see helper_functions.py)
    encodings = encode_responses(tokenizer, responses)
    # embed responses (see helper_functions.py)
    embeddings = embed_responses(model, encodings)
    # reshape embeddings
    num_rows = embeddings.shape[0]
    num_iterations = int(num_rows / num_per_iteration)
    reshaped_embeddings = embeddings.reshape(num_iterations, num_per_iteration, 384)
    # easier to divide with sum than to take mean
    sum_distances = np.zeros((num_per_iteration, num_per_iteration))

    # Loop through the 41 iterations to accumulate the sum of cosine distances for each pair
    for iter_idx in range(num_iterations):
        distances = cosine_distances(reshaped_embeddings[iter_idx, :, :])
        sum_distances += distances

    # Calculate the mean distances
    mean_distances = sum_distances / num_iterations

    # Create the DataFrame to store mean distances
    froms = []
    tos = []
    mean_distances_list = []

    for from_idx in range(num_per_iteration):
        for to_idx in range(num_per_iteration):
            froms.append(from_idx)
            tos.append(to_idx)
            mean_distances_list.append(mean_distances[from_idx, to_idx])

    df_ = pd.DataFrame({
        'from': froms,
        'to': tos,
        'mean_distance': mean_distances_list
    })

    import matplotlib.pyplot as plt 

    heatmap_data = df_.pivot(index="from", 
                            columns="to", 
                            values="mean_distance")

    mask = np.tril(np.ones_like(heatmap_data, dtype=bool), k=-1)

    # Generate the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', cbar=True, mask=mask)
    plt.title(f"{id} {condition}")
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(f'../fig/gpt4/heatmap/{id}_{condition}.png')
