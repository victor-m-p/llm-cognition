'''
VMP 2023-09-11:
This script is used to evaluate the data from GPT-4;
These are helper functions for the analysis, 
e.g. encoding, embedding, and calculating metrics.
'''
from sklearn.decomposition import PCA
import torch
from torch.nn import functional as F

# mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# encode responses
def encode_responses(tokenizer, sentences):
    encoded_input = tokenizer(sentences, 
                              padding=True, 
                              truncation=True, 
                              return_tensors='pt')
    return encoded_input

# embed responses
def embed_responses(model, encoded_input):
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

# pca for n=2 components
def run_PCA(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(embeddings)
    x = transformed[:, 0]
    y = transformed[:, 1]
    return x, y