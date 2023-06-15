'''
https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
testing sBERT embeddings. 
'''

import json 
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd 
import numpy as np 
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
import seaborn as sns 
import matplotlib.pyplot as plt
checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 

model = AutoModel.from_pretrained(checkpoint) # which model?
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open('../gpt/data/text-davinci-003_phillips2017_n10_could_should.json') as user_file:
  phillips2017 = json.load(user_file)

phillips2017
# this is actually a pretty bad setup
contexts = list(phillips2017.keys())
conditions = ['could', 'should']

sentences = [x.values() for x in phillips2017.values()]
flattened = [item for sublist in sentences for value in sublist for item in value]
cleaned = [s.strip() for s in flattened]

encoded_input = tokenizer(cleaned, 
                          padding=True, 
                          truncation=True, 
                          #max_length=128, -- what they train with
                          return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# things we can actually do: 
## 1. avg. diameter per condition (within vignettes)
## 2. avg. diameter per condition (across vignettes)--not sure whether this makes sense
## 3. avg. diameter across conditions
### i.e. even though equally "broad" they might be different
### if say "could" promps something consistent 
### and "should" prompts something consistent (but different)
### just checking whether they are actually different from each other
### basically; we expect "could-could" pairs (on avg.) to be more
### similar than "could-should" pairs. 

# Compute pairwise cosine distances
num_contexts = len(contexts)
step_length = 10 # number of gen. per condition
cosine_list = []
for i in range(0, num_contexts*step_length, step_length):
    cos = cosine_distances(sentence_embeddings[i:i+step_length])
    cosine_list.append(cos)

upper_triangle_list=[]
for array in cosine_list:
    # Get the indices for the upper triangle
    upper_triangle_indices = np.triu_indices(array.shape[0], k = 1)

    # Use these indices to get the upper triangle
    upper_triangle = array[upper_triangle_indices]

    # Convert to list and append to upper_triangles_list
    upper_triangle_list.append(upper_triangle.tolist())

# sort-of useful format but still a bit fucked. 
num_per_group = len(upper_triangle_list[0])
num_contexts = int(len(contexts)/2)

upper_triangle_flat = lst_new = [item for sublist in upper_triangle_list for item in sublist]
context_lst = [item for item in contexts for _ in range(num_per_group)]
conditions_lst = [item for item in conditions for _ in range(num_per_group)]
conditions_lst = conditions_lst * num_contexts

d = pd.DataFrame({
    'contexts': context_lst, 
    'conditions': conditions_lst,
    'cosine_dist': upper_triangle_flat
    })

d['contexts_small'] = d['contexts'].str.split().str[0]

# some tendency for COULD to have a broader
# possibility space than should; but not universally
# true, and we need to run many more to have a very
# certain answer (which we can do...)

# Create a boxplot using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(data=d, 
            x='contexts_small', 
            y='cosine_dist', 
            hue='conditions')

# Set plot labels and title
plt.xlabel('Context')
plt.ylabel('Value')
plt.title('Boxplot by Context and Condition')

# Display the plot
plt.show()

# we should check out how it looks in PCA
# plots as well; that would be pretty cool
# I think. 
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
transformed = pca.fit_transform(sentence_embeddings)

# Extract x and y coordinates
x = transformed[:, 0]
y = transformed[:, 1]

# add to a dataframe
context_lst = [item for item in contexts for _ in range(step_length)]
conditions_lst = [item for item in conditions for _ in range(step_length)]*num_contexts
d = pd.DataFrame({
    'contexts': context_lst,
    'conditions': conditions_lst,
    'x': x,
    'y': y})
d['contexts_small'] = d['contexts'].str.split().str[0]

# Create scatter plot
sns.scatterplot(data=d, 
                x='x', 
                y='y', 
                hue='contexts_small', 
                style='conditions')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')

# Display the plot
plt.show()

# definitely more difference between conditions than modals
# but generally we do seem to get somewhat larger diameter for *could*
# check up on how good the PCA is here.
# makes sense that COULD is still pretty 
# constrained (i.e. constrained by should)
# is the whole point of what they are saying.
# you actually COULD do anything; but you do not.

# can we calculate total span somehow?
# or is that not better than cosine?