'''
VMP 2023-09-11:
This script is used to generate the feature table.
'''

import pandas as pd 
import numpy as np
import scipy.stats as stats
from helper_functions import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances

# setup
eval = pd.read_csv('../data/data_output/gpt4_eval/results.csv')
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
df = df.merge(eval, on=['condition', 'id', 'iteration', 'shuffled'], how='inner')

## compute rank-order ##
dct_rank_order = {}
vignettes = df['id'].unique()
for vignette in vignettes:
    vignette_df = df[df['id'] == vignette]
    N = len(vignette_df)-2
    num_rank = stats.spearmanr(vignette_df['num'], vignette_df['eval'])
    dct_rank_order[vignette] = num_rank[0]

# gather rank-order
df_rank_order = pd.DataFrame.from_dict(dct_rank_order, orient='index', columns=['rank_order'])
df_rank_order['id'] = df_rank_order.index
df_rank_order = df_rank_order.reset_index(drop=True)

## cosine distance of all points ##
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

df['num'] = df['num'] + 1
df['iteration'] = df['iteration'] + 1
id_list = df['id'].unique().tolist()

# Create an empty list to store dictionaries
records = []

# Assuming your id_list and helper functions are defined
for id in id_list:
    responses = df[df['id']==id]['response_clean'].tolist()
    
    # encode sentences (see helper_functions.py)
    encodings = encode_responses(tokenizer, responses)

    # embed responses (see helper_functions.py)
    embeddings = embed_responses(model, encodings)

    # calculate cosine and euclidean distance (see helper_functions.py)
    cosine_dist = cosine_distances(embeddings)
    
    # Get the indices of the upper triangle without the diagonal
    n_ = cosine_dist.shape[0]
    row_idx, col_idx = np.triu_indices(n_, k=1)

    # Extract the values using the indices
    upper_triangle_could = cosine_dist[row_idx, col_idx]

    # Generate a list of dictionaries for the 'could' condition
    records_ = [{'id': id, 'cosine_dist': val} for val in upper_triangle_could.tolist()]
    
    # Add the records to the main list
    records.extend(records_)
    
df_records=pd.DataFrame(records)
df_records_mean=df_records.groupby('id').mean().reset_index()

## cosine order ##
df_list=[]
num_per_iteration=6
for id in id_list:
    # get responses
    responses = df[df['id'] == id]['response_clean'].tolist()
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
        'cosine_distance': cosine_distances_list,
        'id': id
    })
    df_list.append(df_plot)
df_slope=pd.concat(df_list)
mean_val_slope=df_slope.groupby(['num', 'id'])['cosine_distance'].mean().reset_index()

# quick function to calculate slope 
def compute_slope(group):
    n = len(group)
    sum_x = group['num'].sum()
    sum_y = group['cosine_distance'].sum()
    sum_xy = (group['num'] * group['cosine_distance']).sum()
    sum_x_squared = (group['num'] ** 2).sum()

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x_squared - sum_x**2
    
    # Ensure we're not dividing by zero
    if denominator == 0:
        return None
    return numerator / denominator

slopes = mean_val_slope.groupby('id').apply(compute_slope)
slopes = slopes.reset_index(name='slope')

# combine everything;
df_rank_order = df_rank_order.merge(df_records_mean, on='id', how='inner')
df_rank_order = df_rank_order.merge(slopes, on='id', how='inner')
df_rank_order = df_rank_order[['id', 'rank_order', 'cosine_dist', 'slope']]
df_rank_order = df_rank_order.sort_values('slope', ascending=False)

# Convert the DataFrame to LaTeX format
latex_string = df_rank_order.to_latex(index=False)

# Save as a file 
with open('../tables/feature_table.tex', 'w') as file:
    file.write(latex_string)
