'''
used in the paper
'''

import pandas as pd 
import torch
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
from helper_functions import *
from scipy.spatial import ConvexHull
import textwrap 
from transformers import AutoTokenizer, AutoModel

checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
df = pd.read_csv('../data/data_cleaned/gpt4.csv')

# function to select dissimilar points (for labeling)
def label_points(df): 
    selected_indices = np.random.choice(df.index, 6, replace=False)
    responses = []
    total_lines = 0
    for i, index in enumerate(selected_indices, 1):
        x = df['x'].loc[index]
        y = df['y'].loc[index]
        response = df['responses'].loc[index]
        # Insert line breaks for long responses
        max_line_length = 75
        response_lines = textwrap.wrap(response, width=max_line_length)
        #response_lines = [response[j:j+max_line_length] for j in range(0, len(response), max_line_length)]
        responses.append(f"{i}: " + "\n".join(response_lines))
        ax.text(x, y, str(i), fontsize=12, ha='right')

        # Dynamically calculate vertical position
        for line in response_lines:
            plt.figtext(0.1, -0.05 - 0.05 * total_lines, line if line != response_lines[0] else f"{i}: {line}", 
                        fontsize=12, ha="left")
            total_lines += 1

# double check this code 
def equalize_conditions(df):
    # Count unique iterations for each condition
    counts = df.groupby('condition')['iteration'].nunique()
    num_could = counts.get('could', 0)
    num_should = counts.get('should', 0)

    # Determine which condition has more iterations
    largest_condition = 'could' if num_could > num_should else 'should'

    # Calculate the difference in the number of unique iterations
    diff = abs(num_could - num_should)

    # If the number of iterations is the same for both conditions, return the original dataframe
    if diff == 0:
        return df

    # Randomly select iterations to drop from the condition with more iterations
    to_remove = np.random.choice(
        df[df['condition'] == largest_condition]['iteration'].unique(), 
        diff, 
        replace=False
    )

    # Drop the selected iterations and return the modified dataframe
    return df.drop(df[(df['condition'] == largest_condition) & (df['iteration'].isin(to_remove))].index)

# loop over all vignettes
unique_vignettes = df['id'].unique()
n_per_iteration=20
for vignette in unique_vignettes: 

    df_ = df[df['id']==vignette]
    df_ = equalize_conditions(df_)

    # embed and encode
    responses_could = df_[df_['condition']=='could']['response'].tolist()
    responses_should = df_[df_['condition']=='should']['response'].tolist()

    encodings_could = encode_responses(tokenizer, responses_could)
    encodings_should = encode_responses(tokenizer, responses_should)

    embeddings_could = embed_responses(model, encodings_could)
    embeddings_should = embed_responses(model, encodings_should)

    embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
    x_temp, y_temp = run_PCA(embeddings_temp)

    n_per_condition = embeddings_could.shape[0]
    n_iterations = int(n_per_condition/n_per_iteration)

    condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

    df_pca = pd.DataFrame({
        'x': x_temp,
        'y': y_temp,
        'context': ['Mary']*n_per_condition + ['Mary']*n_per_condition,
        'condition': ['could']*n_per_condition + ['should']*n_per_condition,
        'responses': responses_could + responses_should,
        'num': list(range(n_per_iteration)) * n_iterations * 2})

    # plot by num
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.scatterplot(data=df_pca,
                    x=df_pca['x'],
                    y=df_pca['y'],
                    hue='num',
                    alpha=0.5)
                
    label_points(df_pca)
    plt.legend(fontsize=12)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)    
    plt.savefig(f'../fig/gpt4/pca_num/{vignette}.png')
    
    # plot by condition
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.scatterplot(data=df_pca,
                    x=df_pca['x'],
                    y=df_pca['y'],
                    hue='condition',
                    alpha=0.5)
                
    label_points(df_pca)
    plt.legend(fontsize=12)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)    
    plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}.png')