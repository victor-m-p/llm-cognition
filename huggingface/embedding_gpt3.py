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
import seaborn as sns 
import matplotlib.pyplot as plt
checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 

model = AutoModel.from_pretrained(checkpoint) # which model?
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open('../gpt/data/phillips2017_gpt-3.5-turbo_n100_temp0.5.json') as user_file:
    phillips05 = json.load(user_file)

with open('../gpt/data/phillips2017_gpt-3.5-turbo_n100_temp1.0.json') as user_file:
    phillips10 = json.load(user_file)    
    
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

completion_phillips05 = flatten_completions(phillips05)
completion_phillips10 = flatten_completions(phillips10)

# encode responses
def encode_sentences(sentences):
    encoded_input = tokenizer(sentences, 
                              padding=True, 
                              truncation=True, 
                              return_tensors='pt')
    return encoded_input

encoded_phillips05 = encode_sentences(completion_phillips05)
encoded_phillips10 = encode_sentences(completion_phillips10)

# embed responses
def embed_responses(encoded_input):
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

embeddings_phillips05 = embed_responses(encoded_phillips05)
embeddings_phillips10 = embed_responses(encoded_phillips10)

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
    
n_conditions = len(phillips05.keys())
n_generations = len(phillips05[list(phillips05.keys())[0]]['generation'])

cosine_phillips05, pair_phillips05 = calculate_metrics(embeddings_phillips05, n_conditions, n_generations)
cosine_phillips10, pair_phillips10 = calculate_metrics(embeddings_phillips10, n_conditions, n_generations)

# get conditions
n_per_context = int(len(cosine_phillips05)/n_conditions)
def get_conditions(data_dictionary, n_per_context):
    condition_list = [[key for _ in range(n_per_context)] for key in data_dictionary.keys()]
    condition_list = [item for sublist in condition_list for item in sublist]
    return condition_list

conditions_phillips05 = get_conditions(phillips05, n_per_context)
conditions_phillips10 = get_conditions(phillips10, n_per_context)

# gather dataframes 
d_phillips05_within = pd.DataFrame({
    'condition': conditions_phillips05,
    'cosine_dist': cosine_phillips05,
    'pair_dist': pair_phillips05})

d_phillips10_within = pd.DataFrame({
    'condition': conditions_phillips10,
    'cosine_dist': cosine_phillips10,
    'pair_dist': pair_phillips10})

# plot difference within conditions (cosine + pairwise)
sns.boxplot(data=d_phillips05_within,
            x='condition',
            y='cosine_dist')

sns.boxplot(data=d_phillips05_within,
            x='condition',
            y='pair_dist')

sns.boxplot(data=d_phillips10_within,
            x='condition',
            y='cosine_dist')

sns.boxplot(data=d_phillips10_within,
            x='condition',
            y='pair_dist')

# add mean (PWD) for each distance metric 
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

###### between-conditions ######
# do this 

###### PCA #######
def plot_group_PCA(embedding):
        
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
        #style = '-' if cond == 'condition1' else '--'  # adjust according to your conditions
        
        # Draw the lines and fill the areas
        for simplex in hull.simplices:
            plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], color=color)
        plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Plot')

    plt.show() 
    
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

    # Map each unique context to a unique integer
    #unique_contexts = d['context'].unique()
    #context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))

    #for (sent_n, ), group in d.groupby(['context']):
        # Generate ConvexHull for each group
    hull = ConvexHull(d[['x', 'y']])
    
    # Get the color and style based on the group
    color = 'tab:blue'
    #style = '-' if cond == 'condition1' else '--'  # adjust according to your conditions
    
    # Draw the lines and fill the areas
    for simplex in hull.simplices:
        plt.plot(d['x'].iloc[simplex], d['y'].iloc[simplex], color=color)
    plt.fill(d['x'].iloc[hull.vertices], d['y'].iloc[hull.vertices], alpha=0.5, color=color)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Plot')

    plt.show() 

plot_group_PCA(embeddings_phillips05)
plot_group_PCA(embeddings_phillips10)

# test how much is just the word 
list_of_names = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
completion_phillips05_anonymized = completion_phillips05
for name in list_of_names:
    completion_phillips05_anonymized = [s.replace(name, 'X') for s in completion_phillips05_anonymized]

# now try to run the pipeline on this 
encoded_phillips05_anonymized = encode_sentences(completion_phillips05_anonymized)
embedded_phillips05_anonymized = embed_responses(encoded_phillips05_anonymized)
cosine_phillips05_anonymized, pair_phillips05_anonymized = calculate_metrics(embedded_phillips05_anonymized, n_conditions, n_generations)

conditions_phillips05 = get_conditions(phillips05, n_per_context)

# gather dataframes 
d_phillips05_within_anonymized = pd.DataFrame({
    'condition': conditions_phillips05,
    'cosine_dist': cosine_phillips05_anonymized,
    'pair_dist': pair_phillips05_anonymized})

# yes, so this does not really change which makes sense 
sns.boxplot(data=d_phillips05_within_anonymized,
            x='condition',
            y='cosine_dist')

# but this does change a lot 
plot_group_PCA(embedded_phillips05_anonymized)
plot_single_PCA(embedded_phillips05_anonymized[0:100])

# deep dive into one context and the groups of answers provided 
# ... 

# which contexts are most similar to each other?
## we can compare by responses
## and we can compare by the actual prompts
## is this the same? 