'''
Generates raincloud plot used in paper.
'''

import pandas as pd 
import matplotlib.pyplot as plt
import os 
from helper_functions import *
import ptitprince as pt 
from transformers import AutoTokenizer, AutoModel

checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# setup
## reproduce the submitted analysis
temperature='1.0'
inpath='../data/data_output/phillips_chatgpt' # '../data/data_output/phillips_gpt3'
outpath='../fig/phillips_chatgpt' # '../fig/phillips_gpt3'
contexts = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
context_mapping={i: contexts[i] for i in range(len(contexts))}

# match files (see helper_functions.py)
files_could = match_files(os.path.join(inpath, f'*temp{temperature}_could.json'))
files_should = match_files(os.path.join(inpath, f'*temp{temperature}_should.json'))

# load files (see helper_functions.py)
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
encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

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

df_could['condition'] = 'could'
df_should['condition'] = 'should'
df_combined = pd.concat([df_could, df_should])

# raincloud plots 
dx = "context"; dy = "cosine_dist"; ort = "h"; pal = "Set2"; sigma = .05; dhue='condition'
f, ax = plt.subplots(figsize=(5, 7), dpi=300)
ax=pt.RainCloud(x = dx, 
                y = dy, 
                hue = dhue, 
                data = df_combined, 
                palette = pal, 
                bw = sigma,
                width_viol = .7, 
                ax = ax, 
                orient = ort, 
                alpha = .7, 
                dodge = True,
                point_size=0.1,
                box_showfliers=False,
                box_showmeans=True)
plt.xlabel('Cosine Distance')
plt.savefig(os.path.join(outpath, f'fig_png/raincloud_cosine_temp{temperature}.png'), 
            bbox_inches='tight')

dy='euclid_dist'
f, ax = plt.subplots(figsize=(5, 7), dpi=300)
ax=pt.RainCloud(x = dx, 
                y = dy, 
                hue = dhue, 
                data = df_combined, 
                palette = pal, 
                bw = sigma,
                width_viol = .7, 
                ax = ax, 
                orient = ort, 
                alpha = .7, # alpha of distributions (maybe plots)
                dodge = True,
                point_size=0.1,
                box_showfliers=False,
                box_showmeans=True)
plt.xlabel('Euclidean Distance')
plt.savefig(os.path.join(outpath, f'fig_png/raincloud_euclid_temp{temperature}.png'), 
            bbox_inches='tight')
