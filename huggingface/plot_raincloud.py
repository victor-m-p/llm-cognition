'''
Generates raincloud plot used in paper.
'''

import pandas as pd 
import matplotlib.pyplot as plt
import os 
from helper_functions import *
import ptitprince as pt 
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances

checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# setup
## reproduce the submitted analysis
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
id_list = df['id'].unique().tolist()

# Create an empty list to store dictionaries
records = []

# Assuming your id_list and helper functions are defined
for id in id_list:
    responses_could = df[(df['id']==id) & (df['condition'] == 'could')]['response_option'].tolist()
    responses_should = df[(df['id']==id) & (df['condition'] == 'should')]['response_option'].tolist()
    
    # encode sentences (see helper_functions.py)
    encodings_could = encode_responses(tokenizer, responses_could)
    encodings_should = encode_responses(tokenizer, responses_should)

    # embed responses (see helper_functions.py)
    embeddings_could = embed_responses(model, encodings_could)
    embeddings_should = embed_responses(model, encodings_should)

    # calculate cosine and euclidean distance (see helper_functions.py)
    cosine_dist_could = cosine_distances(embeddings_could)
    cosine_dist_should = cosine_distances(embeddings_should)
    
    # Get the indices of the upper triangle without the diagonal
    n_could = cosine_dist_could.shape[0]
    n_should = cosine_dist_should.shape[0]
    row_idx_could, col_idx_could = np.triu_indices(n_could, k=1)
    row_idx_should, col_idx_should = np.triu_indices(n_should, k=1)

    # Extract the values using the indices
    upper_triangle_could = cosine_dist_could[row_idx_could, col_idx_could]
    upper_triangle_should = cosine_dist_should[row_idx_should, col_idx_should]

    # Generate a list of dictionaries for the 'could' condition
    could_records = [{'id': id, 'condition': 'could', 'cosine_dist': val} for val in upper_triangle_could.tolist()]
    
    # Generate a list of dictionaries for the 'should' condition
    should_records = [{'id': id, 'condition': 'should', 'cosine_dist': val} for val in upper_triangle_should.tolist()]
    
    # Calculate cosine distance across conditions
    cosine_dist_across = pairwise_distances(embeddings_could, embeddings_should, metric='cosine')

    # Flatten the 2D array to a 1D array and then to a list
    across_list = cosine_dist_across.flatten().tolist()

    # Generate a list of dictionaries for the 'across' condition
    across_records = [{'id': id, 'condition': 'across', 'cosine_dist': val} for val in across_list]

    # Add the records to the main list
    records.extend(across_records)

    # Add the records to the main list
    records.extend(could_records)
    records.extend(should_records)
    
# Create a DataFrame from the list of dictionaries
result_df = pd.DataFrame(records)

# raincloud plots 
subset_df = result_df[result_df['condition'] != 'across']
dx = "id"; dy = "cosine_dist"; ort = "h"; pal = "Set2"; sigma = .05; dhue='condition'
f, ax = plt.subplots(figsize=(5, 7), dpi=300)
ax=pt.RainCloud(x = dx, 
                y = dy, 
                hue = dhue, 
                data = subset_df, 
                palette = pal, 
                bw = sigma,
                width_viol = .7, 
                ax = ax, 
                orient = ort, 
                alpha = .7, 
                dodge = True,
                point_size=0.1,
                box_showfliers=False,
                box_showmeans=True)
plt.xlabel('Cosine Distance')
plt.savefig('../fig/gpt4/raincloud/cosine_grouped.png', bbox_inches='tight')
plt.close()

dx = "id"; dy = "cosine_dist"; ort = "h"; pal = "Set2"; sigma = .05; dhue='condition'
f, ax = plt.subplots(figsize=(5, 7), dpi=300)
ax=pt.RainCloud(x = dx, 
                y = dy, 
                hue = dhue, 
                data = result_df, 
                palette = pal, 
                bw = sigma,
                width_viol = .7, 
                ax = ax, 
                orient = ort, 
                alpha = .7, 
                dodge = True,
                point_size=0.1,
                box_showfliers=False,
                box_showmeans=True)
plt.xlabel('Cosine Distance')
plt.savefig('../fig/gpt4/raincloud/cosine_grouped_across.png', bbox_inches='tight')

### divergence over could/should and num ###
# Create an empty list to store dictionaries
records = []

# Assuming your helper functions are defined
unique_groups = df.drop_duplicates(subset=['id', 'num'])[['id', 'num']]

for _, group in unique_groups.iterrows():
    id, num = group['id'], group['num']
    
    responses_could = df[(df['id']==id) & (df['num']==num) & (df['condition'] == 'could')]['response_option'].tolist()
    responses_should = df[(df['id']==id) & (df['num']==num) & (df['condition'] == 'should')]['response_option'].tolist()
    
    # encode sentences (see helper_functions.py)
    encodings_could = encode_responses(tokenizer, responses_could)
    encodings_should = encode_responses(tokenizer, responses_should)

    # embed responses (see helper_functions.py)
    embeddings_could = embed_responses(model, encodings_could)
    embeddings_should = embed_responses(model, encodings_should)

    # calculate cosine and euclidean distance (see helper_functions.py)
    cosine_dist_could = cosine_distances(embeddings_could)
    cosine_dist_should = cosine_distances(embeddings_should)
    
    # Get the indices of the upper triangle without the diagonal
    n_could = cosine_dist_could.shape[0]
    n_should = cosine_dist_should.shape[0]
    row_idx_could, col_idx_could = np.triu_indices(n_could, k=1)
    row_idx_should, col_idx_should = np.triu_indices(n_should, k=1)

    # Extract the values using the indices
    upper_triangle_could = cosine_dist_could[row_idx_could, col_idx_could]
    upper_triangle_should = cosine_dist_should[row_idx_should, col_idx_should]

    # Generate a list of dictionaries for the 'could' condition
    could_records = [{'id': id, 'num': num, 'condition': 'could', 'cosine_dist': val} for val in upper_triangle_could.tolist()]
    
    # Generate a list of dictionaries for the 'should' condition
    should_records = [{'id': id, 'num': num, 'condition': 'should', 'cosine_dist': val} for val in upper_triangle_should.tolist()]
    
    # Calculate cosine distance across conditions
    cosine_dist_across = pairwise_distances(embeddings_could, embeddings_should, metric='cosine')

    # Flatten the 2D array to a 1D array and then to a list
    across_list = cosine_dist_across.flatten().tolist()

    # Generate a list of dictionaries for the 'across' condition
    across_records = [{'id': id, 'num': num, 'condition': 'across', 'cosine_dist': val} for val in across_list]
    
    # Add the records to the main list
    records.extend(across_records)

    # Add the records to the main list
    records.extend(could_records)
    records.extend(should_records)

# Create a DataFrame from the list of dictionaries
result_df = pd.DataFrame(records)

# divergence plot;
import seaborn as sns 
across_df = result_df[result_df['condition'] == 'across']
sns.lineplot(data=across_df, x='num', y='cosine_dist', hue='id')
plt.title('Cosine Distance Across Conditions')
plt.savefig('../fig/gpt4/raincloud/cosine_across.png')
plt.close()
could_df = result_df[result_df['condition'] == 'could']
sns.lineplot(data=could_df, x='num', y='cosine_dist', hue='id')
plt.title('Cosine Distance Within Condition: Could')
plt.savefig('../fig/gpt4/raincloud/cosine_could.png')
plt.close()
should_df = result_df[result_df['condition'] == 'should']
plt.title('Cosine Distance Within Condition: Should')
sns.lineplot(data=should_df, x='num', y='cosine_dist', hue='id')
plt.savefig('../fig/gpt4/raincloud/cosine_should.png')
plt.close()