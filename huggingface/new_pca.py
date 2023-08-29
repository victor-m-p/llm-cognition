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
df = pd.read_csv('../data/data_cleaned/gpt4_subset.csv')
n_per_iteration=18

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
        'context': ['Mary']*n_per_condition + ['Mary']*n_per_condition,
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
    

## Mary (early vs. late)
df_ = df[df['id']=='Mary']
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
    'context': ['Mary']*n_per_condition + ['Mary']*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'responses': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='num',
                alpha=0.5)

# split the data in two;

df_early = df_pca[(df_pca['x'] < -0.2) | (df_pca['y'] < -0.3)]
df_late = df_pca[(df_pca['x'] > -0.2) & (df_pca['y'] > -0.3)]

# here we do have a large difference
df_early['num'].mean()
df_late['num'].mean()

pd.set_option('display.max_colwidth', None)
df_early.head(10)

''' more early responses 
explain situation
email copy or picture of homework
asking for help or extension
borrow copy from friend or older sibling
'''

df_late.head(20)

''' more late responses
cry or show distress to get sympathy
try to impress teacher to get partial credit
quick recreation in lunch break
write reflection (very vague)
use personal tech gadgets?
'''

# in genereal there are more specific and
# good responses in the early stuff, and 
# more trash in the late stuff, but it is also
# not super clean. 

## Josh (could vs. should)
df_ = df[df['id']=='Josh']
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
    'context': ['Josh']*n_per_condition + ['Josh']*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'responses': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='condition',
                alpha=0.5)

df_should = df_pca[df_pca['y'] > 0]
df_could = df_pca[df_pca['y'] < 0]

df_should.groupby('condition')['x'].count()
df_could.groupby('condition')['x'].count()

df_should = df_should.sample(frac=1).reset_index(drop=True)
df_should.head(20)

''' more should responses 
inform airport and check possibility of delaying
ensure they can board quickly
fix car himself
research nearby hostels / hotels 
'''

df_could = df_could.sample(frac=1).reset_index(drop=True)
df_could.head(20)

''' more could responses 
put on hazard lights around car
hitchhike to airport
ride-sharing app (e.g. uber) or negotiate with private drivers
hire bicycle
'''

# overall *could* responses try more to actually
# get to the airport, whereas there are more of the
# should responses that give up and start planning
# forward (e.g., hotel for night, inform people, etc.)
# but not super clean cut (two large clusters). 
# also more trash in the "should", e.g. use the 
# idle time productively to plan ahead. 
# there are probably just not a lot of good "should"
# responses in this case and I think this is driving
# the slight divergence. 

# three things:
## 1. qualitative
## 2. across iterations
## 3. could should divergence
## problem: first option and last option have scripts. 
## would be tricky to fix this I think.
## problem: mentioning NAME does A TON!