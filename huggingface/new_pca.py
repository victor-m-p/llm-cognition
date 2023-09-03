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

# setup 
# 1. gpt4.csv and n_per_iteration=20
# 2. gpt4_subset.csv and n_per_iteration=18 (removeing first and last option)
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
n_per_iteration=6

# function to select dissimilar points (for labeling)
def label_points(df): 
    selected_indices = np.random.choice(df.index, 6, replace=False)
    responses = []
    total_lines = 0
    for i, index in enumerate(selected_indices, 1):
        x = df['x'].loc[index]
        y = df['y'].loc[index]
        response = df['response_clean'].loc[index]
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
for vignette in unique_vignettes: 

    df_ = df[df['id']==vignette]
    df_ = equalize_conditions(df_)
    
    # because we are plotting could and should together, we should anonymize
    # otherwise we could be inducing a "false" effect

    # embed and encode
    responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
    responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

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
        'context': [vignette]*n_per_condition + [vignette]*n_per_condition,
        'condition': ['could']*n_per_condition + ['should']*n_per_condition,
        'response_clean': responses_could + responses_should,
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
    plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}.png')
    
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
    plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}_{n_per_iteration}.png')
    

## Liz ##
id = 'Liz'
df_ = df[df['id']==id]
df_ = equalize_conditions(df_)

# embed and encode
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

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
    'context': [id]*n_per_condition + [id]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'responses': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='condition',
                alpha=0.5)

# split the data in two;
df_left_top = df_pca[(df_pca['y'] > 0.15) & 
                     (df_pca['x'] < -0.2)]

df_right_top = df_pca[(df_pca['y'] > -0.1) & 
                      (df_pca['x'] > 0.3)]

main_bulk = df_pca[(df_pca['y'] < 0.2) &
                   (df_pca['x'] > -0.4) & 
                   (df_pca['x'] < 0.25)]

pd.set_option('display.max_colwidth', None)
df_left_top.head(10) # could: do something else (e.g. exercise outside, go back to work)
df_right_top.head(10) # should: renew, promotion, negotiate 
main_bulk.head(10) # free trial, day pass, guest pass (but also some overlap with others)
df_left_top.groupby('condition').count() # 144-9
df_right_top.groupby('condition').count() # 196-29
main_bulk.groupby('condition').count() # almost exactly equal

## Josh ##
id = 'Josh'
df_ = df[df['id']==id]
df_ = equalize_conditions(df_)

# embed and encode
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

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
    'context': [id]*n_per_condition + [id]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'responses': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='condition',
                alpha=0.5)

# split the data in two;
right_top = df_pca[(df_pca['y'] > 0.2) & 
                   (df_pca['x'] > 0.3)]

right_top.groupby('condition').count() # 58-0
right_top.head(10) # mainly about remaining calm, not panicking.

bottom_left = df_pca[(df_pca['y'] < 0.1) & 
                     (df_pca['x'] < -0.2)]

bottom_left.groupby('condition').count() # 260-112
bottom_left.head(10) # all about getting a ride (friend, ride sharing, taxi)

top_batch = df_pca[(df_pca['y'] > 0.4) & 
                   (df_pca['x'] < 0.1)]

top_batch.groupby('condition').count() # majority should but not strong
top_batch.head(10) # all about contacting the airline (reschedule, explain situation, etc.)

## Brad ##
id = 'Brad'
df_ = df[df['id']==id]
df_ = equalize_conditions(df_)

# embed and encode
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

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
    'context': [id]*n_per_condition + [id]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'responses': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='num',
                alpha=0.5)

# split the data
left_mid = df_pca[(df_pca['y'] > -0.1) &
                  (df_pca['y'] < 0.3) & 
                  (df_pca['x'] < -0.2)] 
left_mid['num'].mean() # 1.19 (0-indexed)
left_mid # ration, gather food, conserve energy, assess situation, etc.

# lower cluster
low_mid = df_pca[(df_pca['y'] < 0) & 
                 (df_pca['x'] > -0.3) & 
                 (df_pca['x'] < 0.15)]
low_mid # shelter, warmth, fire (also some overlap with first)
low_mid['num'].mean() # 2.47

# right cluster
right_mid = df_pca[(df_pca['x'] > 0.2) & 
                   (df_pca['y'] > -0.3) & 
                   (df_pca['y'] < 0.2)]
right_mid['num'].mean() # 2.87
right_mid.head(10) # signal for help, create SOS signal, also fire (but here mainly for signal)

# top cluster
top_mid = df_pca[(df_pca['y'] > 0.3) & 
                 (df_pca['x'] > -0.2) & 
                 (df_pca['x'] < 0.2)]

top_mid['num'].mean() # 2.9
top_mid.head(10) # staying together, remain calm, morale, etc.


### try to make the nice plot ###
id = 'Brad'
df_ = df[df['id']==id]
df_ = equalize_conditions(df_)

# embed and encode
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

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
    'context': [id]*n_per_condition + [id]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'responses': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})

n_per_iteration = 6
n_iterations = 100
df_pca['iter'] = (df_pca.index // n_per_iteration) % n_iterations

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='num',
                alpha=0.5)

specific_condition = 'could'
specific_iter = 5

# Filter data for the path line
df_path = df_pca[(df_pca['condition'] == specific_condition) & (df_pca['iter'] == specific_iter)]

# Add path line (in black)
ax.plot(df_path['x'], df_path['y'], color='black', marker='o')

# Highlight start point (in black) and annotate
start_point = df_path[df_path['num'] == 0]
ax.scatter(start_point['x'], start_point['y'], color='black', zorder=5)
ax.annotate('Start', (start_point['x'].values[0], start_point['y'].values[0]), textcoords="offset points", xytext=(0,10), ha='center')

# Show the plot
plt.show()