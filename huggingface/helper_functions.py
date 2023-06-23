from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import numpy as np 
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# encode responses
def encode_responses(sentences):
    encoded_input = tokenizer(sentences, 
                              padding=True, 
                              truncation=True, 
                              return_tensors='pt')
    return encoded_input

def embed_responses(encoded_input):
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def extract_upper_triu(metric_list):
    upper_triangle_list = []
    for metric_group in metric_list:
        upper_triangle_indices = np.triu_indices(metric_group.shape[0], k=1)
        upper_triangle_values = metric_group[upper_triangle_indices]
        upper_triangle_list.append(upper_triangle_values.tolist())
    upper_triangle_list = [item for sublist in upper_triangle_list for item in sublist]
    return upper_triangle_list 

def calculate_metrics(sentence_embeddings, num_ctx, num_gen_total):
    cosine_list = []
    pair_list = []
    for i in range(0, num_ctx*num_gen_total, num_gen_total):
        print(i)
        print(i+num_gen_total)
        # cosine distances
        cos_dist = cosine_distances(sentence_embeddings[i:i+num_gen_total])
        # pairwise distances
        pair_dist = torch.cdist(sentence_embeddings[i:i+num_gen_total], 
                                sentence_embeddings[i:i+num_gen_total])
        # gather metrics
        cosine_list.append(cos_dist)
        pair_list.append(pair_dist)
    cosine_upper_triangle = extract_upper_triu(cosine_list)
    pair_upper_triangle = extract_upper_triu(pair_list)
    return cosine_upper_triangle, pair_upper_triangle

def sort_responses(responses_raw, ctx, num_gen_individual):
    responses_sorted = []
    conditions_sorted = []
    for num_ctx, ctx in enumerate(ctx): 
        for run in responses_raw:
            ele = run[num_ctx*num_gen_individual:num_ctx*num_gen_individual+num_gen_individual]
            responses_sorted.append(ele)
            conditions_sorted.append([ctx for _ in range(len(ele))])
    responses_flattened = [item for sublist in responses_sorted for item in sublist] 
    conditions_flattened = [item for sublist in conditions_sorted for item in sublist]
    return responses_flattened, conditions_flattened

# works with the pairwise distance metrics to sort contexts
def get_pairwise_contexts(contexts, n_per_context):
    context_list = [[key for _ in range(n_per_context)] for key in contexts]
    context_list = [item for sublist in context_list for item in sublist]
    return context_list