'''
investigating the applicability of GMM,
in particular in learning the number of clusters.
'''

# https://scikit-learn.org/stable/modules/mixture.html
# Use BIC to determine the number of clusters

# first the usual preprocessing 


import pandas as pd 
import os 
from helper_functions import *
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.mixture import GaussianMixture
import re

checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# setup
temperature='1.0'
inpath='../data/data_output/vignettes_gpt3' #'../data/data_output/phillips_gpt3'
outpath='../fig/GMM_experiments'

# match files (see helper_functions.py)
files_could = match_files(os.path.join(inpath, f'*temp{temperature}_could.json'))
files_should = match_files(os.path.join(inpath, f'*temp{temperature}_should.json'))

# load files (see helper_functions.py)
# contexts = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
contexts=['Linda', 'Robert', 'James', 'Mary', 'Simon', 'Jack',
          'Maryam', 'Mary', 'Justin', 'Sam', 'Jackson', 'Abraham',
          'Alexandria', 'Emily']

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

# remove gender markers
def remove_gender(sentences, sub_pairs): 
    '''
    Remove gendered words from sentences
    '''
    for old, new in sub_pairs: 
        print(old)
        sentences = [re.sub(old, new, sentence) for sentence in sentences]
    return sentences
sub_pairs=[('He ', 'They '), ('he ', 'they '), 
           ('She ', 'They '), ('she', 'they'), 
           ('Her ', 'Their '), ('her ', 'their '), 
           ('His ', 'Their '), ('his ', 'their '),
           ('him ', 'them '), ('Him ', 'Them '),
           ('hers ', 'theirs '), ('Hers ', 'Theirs '),
           ('himself ', 'themself '), ('herself ', 'themself '),
           ('Himself ', 'Themself '), ('Herself ', 'Themself ')]
responses_could=remove_gender(responses_could, sub_pairs)

# encode sentences (see helper_functions.py)
encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

# GMM 
## across all contexts 
def run_BIC_grid(embeddings, max_components):
    '''
    Run BIC on a range of n_components
    '''
    bic_dict={}
    for n_components in range(max_components): 
        n_components += 1   
        gm = GaussianMixture(n_components=n_components, random_state=0).fit(embeddings)
        bic = gm.bic(embeddings) # 
        bic_dict[n_components]=bic
    return bic_dict

X = embeddings_could #np.array(embeddings_could)
bic_dict=run_BIC_grid(X, 10) #wow, just n=1 (because of n?)
bic_dict
### check the n=3 component model
gm_n2 = GaussianMixture(n_components=2, random_state=0).fit(X)
gm_n4 = GaussianMixture(n_components=4, random_state=0).fit(X)
gm_n6 = GaussianMixture(n_components=6, random_state=0).fit(X)
p2 = gm_n2.fit_predict(X)
p4 = gm_n4.fit_predict(X)
p6 = gm_n6.fit_predict(X)

### check what this captures 
df = pd.DataFrame({
    'context': contexts_could,
    'p2': p2,
    'p4': p4,
    'p6': p6,
})

### ok--so definitely capturing something 
### but funky with n=3 components
df.groupby('p2').context.value_counts()
df.groupby('p4').context.value_counts()
df.groupby('p6').context.value_counts()

## within context (all best for n=1)
n_tot_per_context=num_gen_individual*num_runs
context_index=0
embedding_subset=embeddings_could[context_index*n_tot_per_context:(context_index+1)*n_tot_per_context]
bic_dict=run_BIC_grid(embedding_subset, 10)
bic_dict # n=1 still best. 

## across could-should (all best for n=1)
context_index=5
embeddings_sub_could=embeddings_could[context_index*n_tot_per_context:(context_index+1)*n_tot_per_context]
embeddings_sub_should=embeddings_should[context_index*n_tot_per_context:(context_index+1)*n_tot_per_context]
embeddings_sub=np.concatenate((embeddings_sub_could, embeddings_sub_should), axis=0)
bic_dict=run_BIC_grid(embeddings_sub, 10)
bic_dict