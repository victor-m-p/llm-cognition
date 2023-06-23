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

files_could = match_files(f'../gpt/data/*temp{temperature}_could_fix.json')
files_should = match_files(f'../gpt/data/*temp{temperature}_should_fix.json')

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

# plot PCA across conditions condition: 
def plot_PCA_ctx_hull(df, outname):

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
    #fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(f'fig_png/{outname}.png')
    fig.savefig(f'fig_pdf/{outname}.pdf')
    plt.close()

plot_PCA_ctx_hull(df_pca_could, 
                  f'PCA_hull_temp{temperature}_could_fix')
plot_PCA_ctx_hull(df_pca_should,
                  f'PCA_hull_temp{temperature}_should_fix')

# run PCA on only a single could context
## NB: question is whether it makes more sense to just embed
## once; and then just select subsets. This will give worse
## embedding but will be more slick perhaps?
## I think I will do this for now ...
## we might actually want e.g. Heinz to be consistent color

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

# for running PCA on single context
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
    #fig.suptitle(f'PCA projection of "{context}" context for "{condition}" condition')
    # Adjust layout to account for response labels
    
    plt.tight_layout()
    fig.savefig(f'fig_png/PCA_hull_temp{temperature}_{condition}_{context}_labels_{labels}_fix.png', bbox_inches='tight')
    fig.savefig(f'fig_pdf/PCA_hull_temp{temperature}_{condition}_{context}_labels_{labels}_fix.pdf') 
    plt.close()

# There is one problematic generation (at least) for Brad (temp=0.5)
# Brian is the most interesting for temp=0.5 I think. 

#run_pca 
context_mapping={0: 'Heinz',
                 1: 'Josh',
                 2: 'Brian',
                 3: 'Liz',
                 4: 'Mary',
                 5: 'Brad'}
for context_num, context in context_mapping.items():

    run_PCA_single_context(df_pca_could, 
                           context, 
                           context_num,
                           'could',
                           'true')

# the new concept
def run_PCA_across_condition(df, temperature, context_num, context, fontsize, labels='true', use_hull='true'):
    
    # Map each unique context to a unique integer
    fig, ax = plt.subplots(figsize=(8, 6)) # (8, 6) before
    sns.scatterplot(data=df,
                    x=df['x'],
                    y=df['y'],
                    #color=sns.color_palette()[context_num],
                    hue='condition',
                    alpha=0.5)
    
    # Get the color and style based on the group
    color = sns.color_palette()[context_num]

    if use_hull == 'true': 
        unique_contexts = df['condition'].unique()
        context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))
        # Generate ConvexHull for each group
        hull = ConvexHull(df[['x', 'y']])
        for (cond, ), group in df.groupby(['condition']):
            # Generate ConvexHull for each group
            hull = ConvexHull(group[['x', 'y']])
            
            # Get the color and style based on the group
            color = sns.color_palette()[context_to_int[cond]]
            #style = '-' if cond == 'could' else '--'  # adjust according to your conditions
            
            # Draw the lines and fill the areas
            for simplex in hull.simplices:
                plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], color=color) #style, color=color)
            plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)

    # Label 5 random points
    if labels=='true': 
        #selected_indices = random.sample(range(len(df_context)), 5)
        selected_indices = select_dissimilar_points(df, 6)
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
            ax.text(x, y, str(i), fontsize=fontsize, ha='right')

            # Dynamically calculate vertical position
            for line in response_lines:
                plt.figtext(0.1, -0.05 - 0.05 * total_lines, line if line != response_lines[0] else f"{i}: {line}", 
                            fontsize=12, ha="left")
                total_lines += 1
    
    plt.legend(fontsize=fontsize)
    plt.xlabel('PC1', fontsize=fontsize)
    plt.ylabel('PC2', fontsize=fontsize)
    #fig.suptitle(f'PCA projection of "{context}" context for "{condition}" condition')
    # Adjust layout to account for response labels
    
    plt.tight_layout()
    fig.savefig(f'fig_png/PCA_hull_{use_hull}_temp{temperature}_{context}_labels_{labels}_fix.png', bbox_inches='tight')
    fig.savefig(f'fig_pdf/PCA_hull_{use_hull}_temp{temperature}_{context}_labels_{labels}_fix.pdf') 
    plt.close()

for num, i in enumerate(range(0, num_ctx*num_gen_individual*num_runs, num_gen_individual*num_runs)):
    # embeddings for could and should
    embeddings_could_temp = embeddings_could[i:i+num_gen_individual*num_runs]
    embeddings_should_temp = embeddings_should[i:i+num_gen_individual*num_runs]
    # stack and run pca 
    embeddings_temp = torch.cat((embeddings_could_temp, embeddings_should_temp), 0)
    x_temp, y_temp = run_PCA(embeddings_temp)
    # contexts for could and should
    contexts_could_temp = contexts_could[i:i+num_gen_individual*num_runs]
    contexts_should_temp = contexts_should[i:i+num_gen_individual*num_runs]
    # responses for could and should
    responses_could_temp = responses_could[i:i+num_gen_individual*num_runs]
    responses_should_temp = responses_should[i:i+num_gen_individual*num_runs]
    
    df_pca_temp = pd.DataFrame({
        'x': x_temp,
        'y': y_temp,
        'context': contexts_could_temp + contexts_should_temp,
        'condition': ['could']*num_gen_individual*num_runs + ['should']*num_gen_individual*num_runs,
        'responses': responses_could_temp + responses_should_temp})

    context_temp = context_mapping.get(num)

    run_PCA_across_condition(df_pca_temp,
                             temperature,
                             num,
                             context_temp,
                             18,
                             'true',
                             'false')

# include Phillips data 
# just "could" for now 
with open('data_input/phillips2017.json') as f: 
        phillips2017 = json.load(f)

ctx = list(phillips2017.keys())
ctx_dict = {'context_1': 'Heinz',
            'context_2': 'Josh',
            'context_3': 'Brian',
            'context_4': 'Liz',
            'context_5': 'Mary',
            'context_6': 'Brad'}

ordinary=phillips2017['context_1']['ordinary']
#impossible=phillips2017['context_1']['impossible']
immoral=phillips2017['context_1']['immoral']
#improbable=phillips2017['context_1']['improbable']
#irrational=phillips2017['context_1']['irrational']

# wrangle 
ctx_idx=0
ctx=ctx_dict.get(f'context_{ctx_idx+1}')
phillips2017[f'context_{ctx_idx+1}']['immoral']

responses_could_test=responses_could[0:num_gen_individual*num_runs]
conditions_could_test=['gpt' for i in range(num_gen_individual*num_runs)]
contexts_could_test=[ctx_dict.get(f'context_{ctx_idx+1}') for i in range(num_gen_individual*num_runs)]
keys=['ordinary', 'immoral']
for i in keys:
    response_add=phillips2017[f'context_{ctx_idx+1}'][i]
    context_add=[ctx_dict.get(f'context_{ctx_idx+1}') for i in range(len(response_add))]
    conditions_add=[i for _ in range(len(response_add))]
    responses_could_test=responses_could_test+response_add
    contexts_could_test=contexts_could_test+context_add
    conditions_could_test=conditions_could_test+conditions_add
#responses_could_test=[item for sublist in responses_could_test for item in sublist]
#onditions_could_test=[item for sublist in conditions_could_test for item in sublist]
#contexts_could_test=[item for sublist in contexts_could_test for item in sublist]

# embed 
encodings_could_test = encode_responses(responses_could_test)
embeddings_could_test = embed_responses(encodings_could_test)
x_could_test, y_could_test = run_PCA(embeddings_could_test)

# construct relevant dataframe
df_pca_could_test = pd.DataFrame({
    'x': x_could_test,
    'y': y_could_test,
    'context': contexts_could_test,
    'condition': conditions_could_test,
    'responses': responses_could_test})

# plot this somehow 
def run_PCA_single_context2(df, temperature, condition, labels='true'):

    context=df['context'].unique()[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df,
                    x=df['x'],
                    y=df['y'],
                    hue='condition',
                    alpha=0.5)

    # Label 5 random points
    if labels=='true': 
        #selected_indices = random.sample(range(len(df_context)), 5)
        selected_indices = select_dissimilar_points(df, 6)
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
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.tight_layout()
    fig.savefig(f'fig_png/PCA_phillips_temp{temperature}_{condition}_{context}_labels_{labels}_fix.png', bbox_inches='tight')
    fig.savefig(f'fig_pdf/PCA_phillips_temp{temperature}_{condition}_{context}_labels_{labels}_fix.pdf') 
    plt.close()

run_PCA_single_context2(df_pca_could_test,
                        temperature,
                        'could',
                        'true')
                        
                        

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