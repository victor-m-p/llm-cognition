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

with open('../gpt/data/text-davinci-003_phillips2017_josh_branches.json') as user_file:
  phillips2017 = json.load(user_file)

# this is actually a pretty bad setup
index = [[i for j, completion in enumerate(sublist)] for i, sublist in enumerate(phillips2017)]
index = [item for sublist in index for item in sublist]

sentences = [[completion for completion in sublist] for sublist in phillips2017]
sentences = [item for sublist in sentences for item in sublist]
sentences = [s.strip() for s in sentences]

# add the true sentence
#sentences.append('One thing that Mary could do is to run home to get her homework')
sentences.append('One thing that Josh could do is to reschedule for a later flight')

encoded_input = tokenizer(sentences, 
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

completions_embeddings = sentence_embeddings[:-1]
true_sentence_embedding = sentence_embeddings[-1]

###### difference between true and generations ######
cosine_list = []
pair_list = []
for completion in completions_embeddings:
    # cosine distances
    cos_dist = cosine_distances(completion.reshape(1, 384), 
                                true_sentence_embedding.reshape(1, 384))
    # pairwise distances
    pair_dist = torch.cdist(completion.reshape(1, 384),
                            true_sentence_embedding.reshape(1, 384))
    # gather metrics
    cosine_list.append(cos_dist)
    pair_list.append(pair_dist)

# gather dataframe
d = pd.DataFrame({
    'index': index,
    'cosine_dist': [x[0][0] for x in cosine_list],
    'pair_dist': [x.item() for x in pair_list]})

# plot
# here the first word really does all of the work.
sns.boxplot(data=d,
            x='index',
            y='cosine_dist')

sns.boxplot(data=d,
            x='index',
            y='pair_dist')

####### difference between generations #######
num_contexts = 1
num_words = max(index)+1 # make better setup
step_length = len(phillips2017[0]) # number of gen. per condition
cosine_list = []
pair_list = []
diam_list = []
for i in range(0, num_contexts*num_words*step_length, step_length):
    # cosine distances
    cos_dist = cosine_distances(completions_embeddings[i:i+step_length])
    # pairwise distances
    pair_dist = torch.cdist(completions_embeddings[i:i+step_length], 
                            completions_embeddings[i:i+step_length])
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

cosine_flat = [item for sublist in cosine_list for item in sublist]
pair_flat = [item for sublist in pair_list for item in sublist]
word_num = [item for item in range(num_words) for _ in range(num_per_group)]

#conditions_lst = [item for item in conditions for _ in range(num_per_group)]
#conditions_lst = conditions_lst * num_contexts

d = pd.DataFrame({
    #'context': context_lst, 
    'word_num': word_num,
    'cosine_dist': cosine_flat,
    'pair_dist': pair_flat 
    })

# some tendency for COULD to have a broader
# possibility space than should; but not universally
# true, and we need to run many more to have a very
# certain answer (which we can do...)
def plot_distance(data, x_metric, y_metric, hue_metric): 
    
    # Create a boxplot using seaborn
    plt.figure()
    sns.boxplot(data=data, 
                x=x_metric, 
                y=y_metric) 
                #hue=hue_metric)

    # Set plot labels and title
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f'Boxplot by {x_metric}')#and {hue_metric}')

    # Display the plot
    plt.show()

# these two are basically the same
# in general larger possibility space for could
# but not a super strong effect; and one vignette
# does not show this (with n=10).
# have not done anything with diameter yet
# seems tricky since it bases everything on two points  
plot_distance(d, 'word_num', 'cosine_dist', 'word_num')
plot_distance(d, 'word_num', 'pair_dist', 'word_num')

########## PCA ##########
# Perform PCA
pca = PCA(n_components=2)
transformed = pca.fit_transform(sentence_embeddings)

# Extract x and y coordinates
x = transformed[:, 0]
y = transformed[:, 1]

# add to a dataframe
d = pd.DataFrame({
    'sentence_num': index + [num_words],
    'x': x,
    'y': y})
d['sentence_num'] = d['sentence_num'].astype('category')

# Create scatter plot
from scipy.spatial import ConvexHull

sns.scatterplot(data=d, 
                x='x', 
                y='y', 
                hue='sentence_num') 
                #style='condition')

# Map each unique context to a unique integer
unique_contexts = d['sentence_num'].unique()
context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))

for (sent_n, ), group in d.groupby(['sentence_num']):
    if sent_n<num_words:
        # Generate ConvexHull for each group
        hull = ConvexHull(group[['x', 'y']])
        
        # Get the color and style based on the group
        color = sns.color_palette()[context_to_int[sent_n]]
        #style = '-' if cond == 'condition1' else '--'  # adjust according to your conditions
        
        # Draw the lines and fill the areas
        for simplex in hull.simplices:
            plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], color=color)
        plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)
    else:
        color = sns.color_palette()[context_to_int[sent_n]]
        plt.plot(group['x'], group['y'], color=color, marker='o')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')

plt.show() 