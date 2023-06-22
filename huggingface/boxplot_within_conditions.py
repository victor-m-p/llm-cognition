'''
https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
sBERT embeddings for could prompts.
change condition and temperature for other conditions.
'''

import json 
import pandas as pd 
import torch
from torch.nn import functional as F
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
import glob
from helper_functions import *

# setup
temperature='0.5'

# load files
def match_files(path):
    list_of_files = glob.glob(path)
    list_of_files = sorted(list_of_files)
    return list_of_files

files_could = match_files('../gpt/data/*temp0.5_could*.json')
files_should = match_files('../gpt/data/*temp0.5_should*.json')

# load files
def load_files(list_of_files, list_of_names): 
    completion_superlist = []
    # loop over all files in condition
    for file in list_of_files: 
        # load file
        with open(file) as f: 
            data_dictionary = json.load(f)
        # extract the completions
        completion_list = [[completion for completion in data_dictionary[key]['generation']] for key in data_dictionary.keys()]
        completion_list = [item for sublist in completion_list for item in sublist]
        completion_list = [s.strip() for s in completion_list]
        # if names are mentioned remove them
        for name in list_of_names: 
            completion_list = [s.replace(name, 'X') for s in completion_list]
        # append to generation list 
        completion_superlist.append(completion_list)
    #completion_superlist = [item for sublist in completion_superlist for item in sublist] 
    return completion_superlist

contexts = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
responses_could = load_files(files_could, contexts)
responses_should = load_files(files_should, contexts)

# sort responses to a more useful format 
# assumes that could and should have the same number of contexts
num_ctx = len(contexts)
num_gen_individual=int(len(responses_could[0])/num_ctx) # number of generations per context
num_runs=len(responses_should) # number of times we have run the above
responses_could, _ = sort_responses(responses_could,
                                    contexts,
                                    num_gen_individual)
responses_should, _ = sort_responses(responses_should,
                                     contexts,
                                     num_gen_individual)

# encode sentences (see helper_functions.py)
encodings_could = encode_responses(responses_could)
encodings_should = encode_responses(responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(encodings_could)
embeddings_should = embed_responses(encodings_should)

# calculate cosine and euclidean distance (see helper_functions.py)
num_gen_total=num_gen_individual*num_runs
cosine_dist_could, euclid_dist_could = calculate_metrics(embeddings_could, 
                                                         num_ctx, 
                                                         num_gen_total)
cosine_dist_should, euclid_dist_should = calculate_metrics(embeddings_should,
                                                           num_ctx,
                                                           num_gen_total)

# get conditions
n_per_context = int(len(cosine_dist_could)/num_ctx)
context_list = get_pairwise_contexts(contexts, n_per_context)

# gather dataframes 
df_could = pd.DataFrame({
    'context': context_list,
    'cosine_dist': cosine_dist_could,
    'euclid_dist': euclid_dist_could})
df_should = pd.DataFrame({
    'context': context_list,
    'cosine_dist': cosine_dist_should,
    'euclid_dist': euclid_dist_should})

# plot difference within conditions (cosine + pairwise)
def boxplot_distance_within_conditions(df, distance_metric, distance_label, temperature, condition):
    fig, ax = plt.subplots()
    sns.boxplot(data=df,
                x='context',
                y=distance_metric)
    fig.suptitle(f'{distance_label} Within Contexts')
    plt.xlabel('Context')
    plt.ylabel(distance_label)
    fig.savefig(f'fig/{distance_metric}_within_temp{temperature}_{condition}.png')
    fig.savefig(f'fig/{distance_metric}_within_temp{temperature}_{condition}.pdf')
    plt.close()

boxplot_distance_within_conditions(df_could, 'cosine_dist', 'Cosine Distance', temperature, 'could')
boxplot_distance_within_conditions(df_could, 'euclid_dist', 'Euclidean Distance', temperature, 'could')

# plot difference grouped by condition
df_could['condition'] = 'could'
df_should['condition'] = 'should'
df_combined = pd.concat([df_could, df_should])

def boxplot_distance_across_conditions(df, distance_metric, distance_label, temperature, condition): 
    fig, ax = plt.subplots()
    sns.boxplot(data=df, 
                x='context', 
                y=distance_metric, 
                hue='condition')

    # Set plot labels and title
    plt.xlabel('Context')
    plt.ylabel(distance_label)
    plt.title(f'{distance_label} by Context and Condition')

    fig.savefig(f'fig/{distance_metric}_across_temp{temperature}_{condition}.png')
    fig.savefig(f'fig/{distance_metric}_across_temp{temperature}_{condition}.pdf')
    plt.close()

boxplot_distance_across_conditions(df_combined, 'cosine_dist', 'Cosine Distance', temperature, 'could')
boxplot_distance_across_conditions(df_combined, 'euclid_dist', 'Euclidean Distance', temperature, 'could')

##### that is it then for this one I believe ######


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
