'''
used in the paper 
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
responses_could, _ = sort_responses(responses_could,
                                    contexts,
                                    num_gen_individual)
responses_should, _ = sort_responses(responses_should,
                                     contexts,
                                     num_gen_individual)

# remove one-word responses because these are typically garbage (i.e. "1", where it is because it starts listing and hits full stop)
responses_could = [s for s in responses_could if len(s.split()) > 1]
responses_should = [s for s in responses_should if len(s.split()) > 1]

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
    fig.savefig(f'fig_png/boxplot_{distance_metric}_within_temp{temperature}_{condition}.png')
    fig.savefig(f'fig_pdf/boxplot_{distance_metric}_within_temp{temperature}_{condition}.pdf')
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
    plt.legend(loc='upper right')
    fig.savefig(f'fig_png/boxplot_{distance_metric}_across_temp{temperature}_{condition}.png')
    fig.savefig(f'fig_pdf/boxplot_{distance_metric}_across_temp{temperature}_{condition}.pdf')
    plt.close()

boxplot_distance_across_conditions(df_combined, 'cosine_dist', 'Cosine Distance', temperature, 'could')
boxplot_distance_across_conditions(df_combined, 'euclid_dist', 'Euclidean Distance', temperature, 'could')
