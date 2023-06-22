'''
used in the paper
'''
import random
import json 
import pandas as pd 
import torch
from torch.nn import functional as F
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
import glob
from helper_functions import *
from scipy.spatial import ConvexHull
import textwrap 

# setup
temperature='1.0'

# load files
def match_files(path):
    list_of_files = glob.glob(path)
    list_of_files = sorted(list_of_files)
    return list_of_files

files_could = match_files(f'../gpt/data/*temp{temperature}_could*.json')
files_should = match_files(f'../gpt/data/*temp{temperature}_should*.json')

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
responses_could, contexts_could = sort_responses(responses_could,
                                                 contexts,
                                                 num_gen_individual)
responses_should, contexts_should = sort_responses(responses_should,
                                                   contexts,
                                                   num_gen_individual)

# remove one-word responses because these are typically garbage
#responses_could = [s for s in responses_could if len(s.split()) > 1]
#responses_should = [s for s in responses_should if len(s.split()) > 1]

# encode sentences (see helper_functions.py)
encodings_could = encode_responses(responses_could)
encodings_should = encode_responses(responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(encodings_could)
embeddings_should = embed_responses(encodings_should)

# generate PCA 
def run_PCA(embeddings):
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(embeddings)
    x = transformed[:, 0]
    y = transformed[:, 1]
    return x, y

# run PCA only for could condition
x_could, y_could = run_PCA(embeddings_could)
x_should, y_should = run_PCA(embeddings_should)

# construct relevant dataframe
df_pca_could = pd.DataFrame({
    'x': x_could,
    'y': y_could,
    'context': contexts_could,
    'responses': responses_could})

df_pca_should = pd.DataFrame({
    'x': x_should,
    'y': y_should,
    'context': contexts_should,
    'responses': responses_should})

# plot PCA within condition: 
def plot_PCA_ctx_hull(df, title, outname):

    fig, _ = plt.subplots()
    sns.scatterplot(data=df, 
                    x='x', 
                    y='y', 
                    hue='context',
                    alpha=0.5) 

    # Map each unique context to a unique integer
    unique_contexts = df['context'].unique()
    context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))

    for (sent_n, ), group in df.groupby(['context']):
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
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(f'fig_png/{outname}.png')
    fig.savefig(f'fig_pdf/{outname}.pdf')
    plt.close()

plot_PCA_ctx_hull(df_pca_could, 
                  'PCA projection of "could" condition',
                  f'PCA_hull_temp{temperature}_could')
plot_PCA_ctx_hull(df_pca_should,
                  'PCA projection of "should" condition',
                  f'PCA_hull_temp{temperature}_should')

# run PCA on only a single could context
## NB: question is whether it makes more sense to just embed
## once; and then just select subsets. This will give worse
## embedding but will be more slick perhaps?
## I think I will do this for now ...
## we might actually want e.g. Heinz to be consistent color
context_mapping={0: 'Heinz',
                 1: 'Josh',
                 2: 'Brian',
                 3: 'Liz',
                 4: 'Mary',
                 5: 'Brad'}

# function to select dissimilar points 
def select_dissimilar_points(df, n_points=5):
    # Function to select n points that are furthest apart in terms of Euclidean distance
    selected_indices = []
    mean_point = [np.mean(df['x']), np.mean(df['y'])]
    
    # Select the center point (closest to the mean)
    distances = np.sqrt((df['x'] - mean_point[0])**2 + (df['y'] - mean_point[1])**2)
    selected_indices.append(distances.idxmin())
    
    # Select the remaining points
    for _ in range(1, n_points):
        max_dist = 0
        max_index = None
        for index, row in df.iterrows():
            if index not in selected_indices:
                min_dist = min(np.sqrt((row['x'] - df['x'].loc[selected_indices])**2 + (row['y'] - df['y'].loc[selected_indices])**2))
                if min_dist > max_dist:
                    max_dist = min_dist
                    max_index = index
        selected_indices.append(max_index)
    
    return selected_indices

def run_PCA_single_context(df, context, context_num, condition, labels='false'):
    
    df_context = df[df['context']==context]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_context,
                    x=df_context['x'],
                    y=df_context['y'],
                    color=sns.color_palette()[context_num],
                    alpha=0.5)
    
    # Generate ConvexHull for each group
    hull = ConvexHull(df_context[['x', 'y']])
    
    # Get the color and style based on the group
    color = sns.color_palette()[context_num]

    # Draw the lines and fill the areas
    for simplex in hull.simplices:
        plt.plot(df_context['x'].iloc[simplex], df_context['y'].iloc[simplex], color=color)
    plt.fill(df_context['x'].iloc[hull.vertices], df_context['y'].iloc[hull.vertices], alpha=0.5, color=color)
    
    # Label 5 random points
    if labels=='true': 
        #selected_indices = random.sample(range(len(df_context)), 5)
        selected_indices = select_dissimilar_points(df_context, 6)
        responses = []
        total_lines = 0
        for i, index in enumerate(selected_indices, 1):
            x = df_context['x'].loc[index]
            y = df_context['y'].loc[index]
            response = df_context['responses'].loc[index]
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
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    fig.suptitle(f'PCA projection of "{context}" context for "{condition}" condition')
    # Adjust layout to account for response labels
    
    plt.tight_layout()
    fig.savefig(f'fig_png/PCA_hull_temp{temperature}_{condition}_{context}_labels_{labels}.png', bbox_inches='tight')
    fig.savefig(f'fig_pdf/PCA_hull_temp{temperature}_{condition}_{context}_labels_{labels}.pdf') 
    plt.close()

# There is one problematic generation (at least) for Brad (temp=0.5)
# Brian is the most interesting for temp=0.5 I think. 
for context_num, context in context_mapping.items():
    run_PCA_single_context(df_pca_could, 
                           context, 
                           context_num,
                           'could',
                           'true')

# run PCA across conditions
# merge the embeddings
embeddings_grouped = torch.cat((embeddings_could, embeddings_should), 0)
x_grouped, y_grouped = run_PCA(embeddings_grouped)

df_pca_grouped = pd.DataFrame({
    'x': x_grouped,
    'y': y_grouped,
    'context': contexts_could + contexts_should,
    'condition': ['could']*len(contexts_could) + ['should']*len(contexts_should)
})

# plot PCA across conditions
def plot_PCA_ctx_cond_hull(df, title, outname):

    # Map each unique context to a unique integer
    unique_contexts = df['context'].unique()
    context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, 
                    x='x', 
                    y='y', 
                    hue='context', 
                    style='condition',
                    alpha=0.5)

    for (ctxt, cond), group in df.groupby(['context', 'condition']):
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
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(f'fig_png/{outname}.png')
    fig.savefig(f'fig_pdf/{outname}.pdf')
    plt.close()
    
plot_PCA_ctx_cond_hull(df_pca_grouped, 
                       'PCA projection across conditions', 
                       f'PCA_hull_temp{temperature}_grouped')

# (add Jonathan Phillips generated possibilities) 