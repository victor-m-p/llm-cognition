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
temperature='1.0'
condition='batched'
file_name=f'phillips2017_{model_name}_n{n}_temp{temperature}_{condition}.json'
full_path = os.path.join(file_path, file_name)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open(full_path) as user_file:
    data_could = json.load(user_file)

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

completions_could = flatten_completions(data_could)

# remove names  
list_of_names = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
for name in list_of_names:
    completions_could = [s.replace(name, 'X') for s in completions_could]

# encode responses
def encode_sentences(sentences):
    encoded_input = tokenizer(sentences, 
                              padding=True, 
                              truncation=True, 
                              return_tensors='pt')
    return encoded_input

encodings_could = encode_sentences(completions_could)

# embed responses
def embed_responses(encoded_input):
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

embeddings_could = embed_responses(encodings_could)

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
    
n_contexts= len(data_could.keys())
n_generations = len(data_could[list(data_could.keys())[0]]['generation'])

cosine_dist_could, euclid_dist_could = calculate_metrics(embeddings_could, n_contexts, n_generations)

# get conditions
n_per_context = int(len(cosine_dist_could)/n_contexts)
def get_contexts(data_dictionary, n_per_context):
    context_list = [[key for _ in range(n_per_context)] for key in data_dictionary.keys()]
    context_list = [item for sublist in context_list for item in sublist]
    return context_list

contexts_could = get_contexts(data_could, n_per_context)

# gather dataframes 
df_could = pd.DataFrame({
    'context': contexts_could,
    'cosine_dist': cosine_dist_could,
    'euclid_dist': euclid_dist_could})

# plot difference within conditions (cosine + pairwise)
fig, ax = plt.subplots()
sns.boxplot(data=df_could,
            x='context',
            y='cosine_dist')
fig.suptitle('Cosine Distance Within Contexts')
fig.savefig(f'fig/cosine_within_{model_name}_n{n}_temp{temperature}_{condition}.png')
plt.close()

fig, ax = plt.subplots()
sns.boxplot(data=df_could,
            x='context',
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
def plot_context_PCA(embedding, n_contexts, outname):
        
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(embedding)

    # Extract x and y coordinates
    x = transformed[:, 0]
    y = transformed[:, 1]

    # add to a dataframe
    d = pd.DataFrame({
        'context': [f'context_{c}' for c in range(n_contexts) for _ in range(n_generations)],
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

plot_context_PCA(embeddings_could, 
                 n_contexts, 
                 f'fig/PCA_{model_name}_n{n}_temp{temperature}_{condition}.png')

# deep dive into one context and the groups of answers provided 
# ... 

# which contexts are most similar to each other?
## we can compare by responses
## and we can compare by the actual prompts
## is this the same? 

# check against the "SHOULD" data 
file_name=f'phillips2017_{model_name}_n{n}_temp{temperature}_should_{condition}.json'
full_path = os.path.join(file_path, file_name)

with open(full_path) as user_file:
    data_should = json.load(user_file)

# run the pipeline
completions_should = flatten_completions(data_should)
for name in list_of_names:
    completions_should = [s.replace(name, 'X') for s in completions_should]
encodings_should = encode_sentences(completions_should)
embeddings_should = embed_responses(encodings_should)
cosine_dist_should, euclid_dist_should = calculate_metrics(embeddings_should, n_contexts, n_generations)
contexts_should = get_contexts(data_should, n_per_context)
df_should = pd.DataFrame({
    'context': contexts_should,
    'cosine_dist': cosine_dist_should,
    'euclid_dist': euclid_dist_should})

df_should['condition'] = 'should'
df_could['condition'] = 'could'

df_combined = pd.concat([df_could, df_should])

def plot_distance(d, dist_metric): 
    
    # Create a boxplot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=d, 
                x='context', 
                y=dist_metric, 
                hue='condition')

    # Set plot labels and title
    plt.xlabel('Context')
    plt.ylabel(dist_metric)
    plt.title('Boxplot by Context and Condition')

    # Display the plot
    plt.show()

# consistent but very weak effect 
plot_distance(df_combined, 'cosine_dist')
plot_distance(df_combined, 'euclid_dist')

# merge the two PCA dimensions
embeddings_grouped = torch.cat((embeddings_could, embeddings_should), 0)
completions_grouped = completions_could + completions_should
def construct_PCA_dataframe(embedding, completions, n_contexts, n_generations): 
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(embedding)

    # Extract x and y coordinates
    x = transformed[:, 0]
    y = transformed[:, 1]

    # add to a dataframe
    context_lst = [f'context_{c}' for c in range(n_contexts) for _ in range(n_generations)]*2
    conditions_lst = [item for item in ['could', 'should'] for _ in range(n_contexts*n_generations)]
    d = pd.DataFrame({
        'context': context_lst,
        'condition': conditions_lst,
        'completions': completions,
        'x': x,
        'y': y})
    return d 

d_PCA_grouped = construct_PCA_dataframe(embeddings_grouped,
                                        completions_grouped,
                                        n_contexts, 
                                        n_generations)

# find points that are contained within several contexts 
# not sure how to do this in a very good way
# for now doing it in a non-good way 
ctx2_ctx3_overlap = d_PCA_grouped[(d_PCA_grouped['x'] > 0) & 
                                  (d_PCA_grouped['x'] < 0.1) &
                                  (d_PCA_grouped['y'] > 0.1) & 
                                  (d_PCA_grouped['y'] < 0.3)]

pd.set_option('display.max_colwidth', None)
ctx2_ctx3_overlap


def PCA_grouped(embedding, n_contexts, n_generations):
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(embedding)

    # Extract x and y coordinates
    x = transformed[:, 0]
    y = transformed[:, 1]

    # add to a dataframe
    context_lst = [f'context_{c}' for c in range(n_contexts) for _ in range(n_generations)]*2
    conditions_lst = [item for item in ['could', 'should'] for _ in range(n_contexts*n_generations)]
    d = pd.DataFrame({
        'context': context_lst,
        'condition': conditions_lst,
        'x': x,
        'y': y})
    d['context'] = d['context'].str.split().str[0]

    # Create scatter plot
    from scipy.spatial import ConvexHull

    sns.scatterplot(data=d, 
                    x='x', 
                    y='y', 
                    hue='context', 
                    style='condition')

    # Map each unique context to a unique integer
    unique_contexts = d['context'].unique()
    context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))

    for (ctxt, cond), group in d.groupby(['context', 'condition']):
        # Generate ConvexHull for each group
        hull = ConvexHull(group[['x', 'y']])
        
        # Get the color and style based on the group
        color = sns.color_palette()[context_to_int[ctxt]]
        style = '-' if cond == 'could' else '--'  # adjust according to your conditions
        
        # Draw the lines and fill the areas
        for simplex in hull.simplices:
            plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], style, color=color)
        plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Plot')

    plt.show() 

PCA_grouped(embeddings_grouped, n_contexts, n_generations)

## consider doing some distance across could-should
## consider doing something across conditions
# i.e. this will tell us about specificity of conditions
# it will tell us whether there is overlap of could/should.
# consider whether these vignettes are morally salient enough
# remember that we also have our own set of vignettes which 
# are all morally salient.
# there is no super strong reason to use phillips perhaps. 


# largest difference in possibility space between 
# context 5 (wide) and context 4 (narrow).
# is this reasonable?

# take context_0 for instance
# what are some could that are not in should

# what is the place in the middle?
# is it just very generic?
