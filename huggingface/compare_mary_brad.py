'''
not used in current manuscript
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
import ptitprince as pt 

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
responses_could, _ = sort_responses(responses_could,
                                    contexts,
                                    num_gen_individual)
responses_should, _ = sort_responses(responses_should,
                                     contexts,
                                     num_gen_individual)

# remove one-word responses because these are typically garbage (i.e. "1", where it is because it starts listing and hits full stop)
#responses_could = [s for s in responses_could if len(s.split()) > 1]
#responses_should = [s for s in responses_should if len(s.split()) > 1]

# encode sentences (see helper_functions.py)
encodings_could = encode_responses(responses_could)
encodings_should = encode_responses(responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(encodings_could)
embeddings_should = embed_responses(encodings_should)

num_gen_total=num_gen_individual*num_runs
cosine_dist_could, euclid_dist_could = calculate_metrics(embeddings_could, 
                                                         num_ctx, 
                                                         num_gen_total)
cosine_dist_should, euclid_dist_should = calculate_metrics(embeddings_should,
                                                           num_ctx,
                                                           num_gen_total)

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

df_could.groupby('context').describe()
df_should.groupby('context').describe()

### now do it for the new cases
files_mary_brad = match_files(f'../gpt/data/mary*temp{temperature}_could.json')
contexts = ['Mary', 'Brad']
responses_mary_brad = load_files(files_mary_brad, contexts)
responses_mary_brad
lst=[]
responses_mary_brad
for i in responses_mary_brad:
    mary=i[0:100]
    lst.append(mary)
for i in responses_mary_brad: 
    brad=i[100:200]
    lst.append(brad)
flat_responses=[item for sublist in lst for item in sublist]

# encode sentences (see helper_functions.py)
encodings_mary_brad = encode_responses(flat_responses)
embeddings_mary_brad = embed_responses(encodings_mary_brad)

num_gen_individual
num_runs=2
num_ctx=2
num_gen_total=num_gen_individual*num_runs
cosine_mary_brad, euclid_mary_brad = calculate_metrics(embeddings_mary_brad, 
                                                         num_ctx, 
                                                         num_gen_total)

n_per_context = int(len(cosine_mary_brad)/num_ctx)
context_list = get_pairwise_contexts(contexts, n_per_context)

df_mary_brad=pd.DataFrame({
    'context': context_list,
    'cosine_dist': cosine_mary_brad,
    'euclid_dist': euclid_mary_brad})

df_mary_brad.groupby('context').describe() # brad context still slightly "larger" but only by a smidgeon
df_could.groupby('context').describe()

vector1_reshaped = embeddings_mary_brad[0:200]
vector2_reshaped = embeddings_mary_brad[200:400]

lst=[]
distances1 = torch.cdist(vector2_reshaped, vector2_reshaped)
distances2 = torch.cdist(vector1_reshaped, vector1_reshaped)
lst.append(distances1)
lst.append(distances2)

upper_triangle_list=[]
for i in lst: 
    upper_triangle_indices = np.triu_indices(i.shape[0], k=1)
    upper_triangle_values = i[upper_triangle_indices]
    upper_triangle_list.append(upper_triangle_values.tolist())
upper_triangle_list = [item for sublist in upper_triangle_list for item in sublist]

dftest=pd.DataFrame({
    'context': context_list,
    'upper_triangle': upper_triangle_list})
dftest.groupby('context').describe()

df_mary_brad.groupby('context').describe()

#upper_triangle_indices = np.triu_indices(distances.shape[0], k=1)
#upper_triangle_values = distances[upper_triangle_indices]
#upper_dist = extract_upper_triu(distances)
#distances = distances.squeeze(0)

np.median(upper_triangle_values) 
df_mary_brad.groupby('context').median()
# 0.844 (vector 1)
# 1.15 (vector 2)
