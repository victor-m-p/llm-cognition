'''
https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
testing sBERT embeddings. 
'''

from sklearn.decomposition import PCA
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

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open('../gpt/data/text-davinci-003_phillips2017_n10_could_should.json') as user_file:
  phillips2017 = json.load(user_file)

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

## 1. avg. diameter per condition (within vignettes)
## 2. avg. diameter per condition (across vignettes)--not sure whether this makes sense
## 3. avg. diameter across conditions
### i.e. even though equally "broad" they might be different
### if say "could" promps something consistent 
### and "should" prompts something consistent (but different)
### just checking whether they are actually different from each other
### basically; we expect "could-could" pairs (on avg.) to be more
### similar than "could-should" pairs. 

#do we need something like this?
#average_pairwise_distance = torch.sum(pairwise_distances) / (pairwise_distances.numel() - embeddings.shape[0])

# Compute pairwise cosine distances
num_contexts = len(contexts)
step_length = 10 # number of gen. per condition
cosine_list = []
pair_list = []
diam_list = []
for i in range(0, num_contexts*step_length, step_length):
    # cosine distances
    cos_dist = cosine_distances(sentence_embeddings[i:i+step_length])
    # pairwise distances
    pair_dist = torch.cdist(sentence_embeddings[i:i+step_length], 
                            sentence_embeddings[i:i+step_length])
    # diameter
    diameter = torch.max(pair_dist)
    diam_list.append(diameter)
    # gather metrics
    cosine_list.append(cos_dist)
    pair_list.append(pair_dist)

def extract_upper_triu(metric_list):
    upper_triangle_list = []
    for metric_group in metric_list:
        upper_triangle_indices = np.triu_indices(metric_group.shape[0], k=1)
        upper_triangle_values = metric_group[upper_triangle_indices]
        upper_triangle_list.append(upper_triangle_values.tolist())
    return upper_triangle_list 

cosine_list = extract_upper_triu(cosine_list)
pair_list = extract_upper_triu(pair_list)

# sort-of useful format but still a bit fucked. 
num_per_group = len(cosine_list[0])
num_contexts = int(len(contexts)/2)

cosine_flat = [item for sublist in cosine_list for item in sublist]
pair_flat = [item for sublist in pair_list for item in sublist]
context_lst = [item for item in contexts for _ in range(num_per_group)]
conditions_lst = [item for item in conditions for _ in range(num_per_group)]
conditions_lst = conditions_lst * num_contexts

d = pd.DataFrame({
    'context': context_lst, 
    'condition': conditions_lst,
    'cosine_dist': cosine_flat,
    'pair_dist': pair_flat 
    })

d['context'] = d['context'].str.split().str[0]

# some tendency for COULD to have a broader
# possibility space than should; but not universally
# true, and we need to run many more to have a very
# certain answer (which we can do...)
def plot_distance(dist_metric): 
    
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

# these two are basically the same
# in general larger possibility space for could
# but not a super strong effect; and one vignette
# does not show this (with n=10).
# have not done anything with diameter yet
# seems tricky since it bases everything on two points  
plot_distance('cosine_dist')
plot_distance('pair_dist')

########## PCA ##########
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
    style = '-' if cond == 'condition1' else '--'  # adjust according to your conditions
    
    # Draw the lines and fill the areas
    for simplex in hull.simplices:
        plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], style, color=color)
    plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')

plt.show() 

# definitely more difference between conditions than modals
# but generally we do seem to get somewhat larger diameter for *could*
# check up on how good the PCA is here.
# makes sense that COULD is still pretty 
# constrained (i.e. constrained by should)
# is the whole point of what they are saying.
# you actually COULD do anything; but you do not.

### calculating span ###

# now we should try to do the things that Simon wants to do
# i.e. look for surprise-patterns within the generations.
# is e.g. "could" more surprising than "should"?
# need to run it through some pipeline to do this
# probably need to rely on gpt2 for now. 