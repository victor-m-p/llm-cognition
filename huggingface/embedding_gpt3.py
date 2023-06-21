'''
https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
testing sBERT embeddings. 
'''

from sklearn.decomposition import PCA
import json 
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd 
import numpy as np 
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
import os 
import seaborn as sns 
import matplotlib.pyplot as plt
checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 

model = AutoModel.from_pretrained(checkpoint) # which model?
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# setup
file_path = '../gpt/data'
model_name = 'text-davinci-003'
n='100'
temperature='0.5'
condition='batched'
file_name=f'phillips2017_{model_name}_n{n}_temp{temperature}_{condition}.json'
full_path = os.path.join(file_path, file_name)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open(full_path) as user_file:
    generation_data = json.load(user_file)

# observations
## really repetitive
## seems like there are patterns
## try to generate this one at a time

# flatten responses
def flatten_completions(data_dictionary): 
    completion_list = [[completion for completion in data_dictionary[key]['generation']] for key in data_dictionary.keys()]
    completion_list = [item for sublist in completion_list for item in sublist]
    completion_list = [s.strip() for s in completion_list]
    return completion_list

completions = flatten_completions(generation_data)

# remove names  
list_of_names = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
for name in list_of_names:
    completions = [s.replace(name, 'X') for s in completions]

# encode responses
def encode_sentences(sentences):
    encoded_input = tokenizer(sentences, 
                              padding=True, 
                              truncation=True, 
                              return_tensors='pt')
    return encoded_input

encodings = encode_sentences(completions)

# embed responses
def embed_responses(encoded_input):
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

embeddings = embed_responses(encodings)

def extract_upper_triu(metric_list):
    upper_triangle_list = []
    for metric_group in metric_list:
        upper_triangle_indices = np.triu_indices(metric_group.shape[0], k=1)
        upper_triangle_values = metric_group[upper_triangle_indices]
        upper_triangle_list.append(upper_triangle_values.tolist())
    upper_triangle_list = [item for sublist in upper_triangle_list for item in sublist]
    return upper_triangle_list 

def calculate_metrics(sentence_embeddings, n_conditions, n_generations):
    cosine_list = []
    pair_list = []
    #diam_list = []
    for i in range(0, n_conditions*n_generations, n_generations):
        # cosine distances
        cos_dist = cosine_distances(sentence_embeddings[i:i+n_generations])
        # pairwise distances
        pair_dist = torch.cdist(sentence_embeddings[i:i+n_generations], 
                                sentence_embeddings[i:i+n_generations])
        # diameter
        #diameter = torch.max(pair_dist)
        #diam_list.append(diameter)
        # gather metrics
        cosine_list.append(cos_dist)
        pair_list.append(pair_dist)
    cosine_upper_triangle = extract_upper_triu(cosine_list)
    pair_upper_triangle = extract_upper_triu(pair_list)
    return cosine_upper_triangle, pair_upper_triangle
    
n_conditions = len(generation_data.keys())
n_generations = len(generation_data[list(generation_data.keys())[0]]['generation'])

cosine_dist, euclid_dist = calculate_metrics(embeddings, n_conditions, n_generations)

# get conditions
n_per_context = int(len(cosine_dist)/n_conditions)
def get_conditions(data_dictionary, n_per_context):
    condition_list = [[key for _ in range(n_per_context)] for key in data_dictionary.keys()]
    condition_list = [item for sublist in condition_list for item in sublist]
    return condition_list

conditions = get_conditions(generation_data, n_per_context)

# gather dataframes 
df_distance = pd.DataFrame({
    'condition': conditions,
    'cosine_dist': cosine_dist,
    'euclid_dist': euclid_dist})

# plot difference within conditions (cosine + pairwise)
fig, ax = plt.subplots()
sns.boxplot(data=df_distance,
            x='condition',
            y='cosine_dist')
fig.suptitle('Cosine Distance Within Conditions')
fig.savefig(f'fig/cosine_within_{model_name}_n{n}_temp{temperature}_{condition}.png')
plt.close()

fig, ax = plt.subplots()
sns.boxplot(data=df_distance,
            x='condition',
            y='euclid_dist')
fig.suptitle('Euclidean Distance Within Conditions')
fig.savefig(f'fig/euclid_within_{model_name}_n{n}_temp{temperature}_{condition}.png')
plt.close()

# add mean (PWD) for each distance metric 
'''
def calculate_PWD(dataframe, metric):
    pass 

def calculate_C(dataframe, metric): 
    pass     

d_phillips05_cosine_mean = d_phillips05_within.groupby(['condition'])['cosine_dist'].mean().reset_index(name='cosine_mean')
d_phillips05_pair_mean = d_phillips05_within.groupby(['condition'])['pair_dist'].mean().reset_index(name='pair_mean')

# difference from mean (PWD) for each distance metric
d_phillips05_within = pd.merge(d_phillips05_within, d_phillips05_cosine_mean, on='condition')
d_phillips05_within = pd.merge(d_phillips05_within, d_phillips05_pair_mean, on='condition')
d_phillips05_within['cosine_subtract_square'] = (d_phillips05_within['cosine_dist'] - d_phillips05_within['cosine_mean'])**2
d_phillips05_within['pair_subtract_square'] = (d_phillips05_within['pair_dist'] - d_phillips05_within['pair_mean'])**2

# calculate C for each condition
d_phillips05_mean_cos_subtract_square = d_phillips05_within.groupby('condition')['cosine_subtract_square'].mean().reset_index(name='mean_cos_subtract_square')
d_phillips05_mean_cos_subtract_square['power'] = d_phillips05_mean_cos_subtract_square['mean_cos_subtract_square']**0.5
d_phillips05_mean_cos_subtract_square = d_phillips05_cosine_mean.merge(d_phillips05_mean_cos_subtract_square)
d_phillips05_mean_cos_subtract_square['C'] = d_phillips05_mean_cos_subtract_square['power']/d_phillips05_mean_cos_subtract_square['cosine_mean']
'''

###### between-conditions ######
# do this 

###### PCA #######
def plot_group_PCA(embedding, outname):
        
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(embedding)

    # Extract x and y coordinates
    x = transformed[:, 0]
    y = transformed[:, 1]

    # add to a dataframe
    d = pd.DataFrame({
        'context': [f'condition_{c}' for c in range(n_conditions) for _ in range(n_generations)],
        'x': x,
        'y': y})

    # Create scatter plot
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots()
    sns.scatterplot(data=d, 
                    x='x', 
                    y='y', 
                    hue='context') 

    # Map each unique context to a unique integer
    unique_contexts = d['context'].unique()
    context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))

    for (sent_n, ), group in d.groupby(['context']):
        # Generate ConvexHull for each group
        hull = ConvexHull(group[['x', 'y']])
        
        # Get the color and style based on the group
        color = sns.color_palette()[context_to_int[sent_n]]

        # Draw the lines and fill the areas
        for simplex in hull.simplices:
            plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], color=color)
        plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    fig.suptitle('PCA embeddings')
    fig.savefig(f'{outname}')
    plt.close()
    
def plot_single_PCA(embedding):
    
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(embedding)

    # Extract x and y coordinates
    x = transformed[:, 0]
    y = transformed[:, 1]

    # add to a dataframe
    d = pd.DataFrame({
        'x': x,
        'y': y})

    # Create scatter plot
    from scipy.spatial import ConvexHull

    sns.scatterplot(data=d, 
                    x='x', 
                    y='y') 
    hull = ConvexHull(d[['x', 'y']])
    color = 'tab:blue'
    # Draw the lines and fill the areas
    for simplex in hull.simplices:
        plt.plot(d['x'].iloc[simplex], d['y'].iloc[simplex], color=color)
    plt.fill(d['x'].iloc[hull.vertices], d['y'].iloc[hull.vertices], alpha=0.5, color=color)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Plot')
    plt.show() 

plot_group_PCA(embeddings, f'fig/PCA_{model_name}_n{n}_temp{temperature}_{condition}.png')

# deep dive into one context and the groups of answers provided 
# ... 

# which contexts are most similar to each other?
## we can compare by responses
## and we can compare by the actual prompts
## is this the same? 