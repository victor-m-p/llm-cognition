import json
import pandas as pd 
import re 
import os 
import glob

inpath = '../data/data_output/gpt4/*.json'
filenames = glob.glob(inpath)

def filter_func(x, num_per_iteration=20):
    return len(x) == num_per_iteration 

df_list = []
for file in filenames:
    # Load the JSON file
    with open(file) as f:
        dct = json.load(f)
    
    # Initialize empty lists to hold DataFrame values
    condition = re.search(r'\d+_p\d+\.\d+_(.*?)\.json', file).group(1)
    ids = []
    vignettes = []
    prompts = []
    iterations = []
    nums = []
    responses = []

    # Loop through dictionary
    for context, values in dct.items():
        id_ = values['id']
        vignette = values['vignette']
        prompt = values['prompt']
        generations = values['generation']
        
        iteration = 1  # Counter for the iteration
        
        for generation in generations:
            # Split generation into individual responses based on the numbering
            individual_responses = re.split(r'\d+\.', generation)
            
            # Remove any empty strings
            individual_responses = [x.strip() for x in individual_responses if x.strip()]
            
            for num, response in enumerate(individual_responses, 1):
                ids.append(id_)
                vignettes.append(vignette)
                prompts.append(prompt)
                iterations.append(iteration)
                nums.append(num)
                responses.append(response)
            
            iteration += 1  # Increment the iteration counter

    # Create the DataFrame
    df = pd.DataFrame({
        'condition': condition,  
        'id': ids,
        'vignette': vignettes,
        'prompt': prompts,
        'iteration': iterations,
        'num': nums,
        'response': responses
    })
    
    # Only iterations that are complete
    df = df.groupby(['iteration']).filter(filter_func)
    
    # Append to list of DataFrames
    df_list.append(df)

# Concatenate all DataFrames
df = pd.concat(df_list)
df.to_csv('../data/data_cleaned/gpt4.csv', index=False)








len(df)
pd.set_option('display.max_colwidth', None)
df.groupby(['num']).count()
df.groupby(['iteration']).count()

# remove iterations that somehow failed
def filter_func(x):
    return len(x) == 20 

df = df.groupby(['iteration']).filter(filter_func)

# (here we will save)

### start analyzing ###
checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
from helper_functions import *

# then we sort by iteration and num
df = df.sort_values(['iteration', 'num'])

# then we take responses out 
responses = df['response'].tolist()

# encode sentences (see helper_functions.py)
encodings = encode_responses(tokenizer, responses)

# embed responses (see helper_functions.py)
embeddings = embed_responses(model, encodings)

### 1. how similar is each first, second, n element to each other ###
from sklearn.metrics.pairwise import cosine_distances
num_per_iteration = 20
num_rows = embeddings.shape[0]
num_iterations = int(num_rows / num_per_iteration)
reshaped_embeddings = embeddings.reshape(num_iterations, num_per_iteration, 384)
mean_cosine_distances = []
nums = []
froms = []
tos = []
cosine_distances_list = []
for i in range(num_per_iteration):
    col_elements = reshaped_embeddings[:, i, :]
    distances = cosine_distances(col_elements)
    
    for from_idx in range(len(distances)):
        for to_idx in range(len(distances)):
            nums.append(i+1) # +1 just for prettier plot
            froms.append(from_idx)
            tos.append(to_idx)
            cosine_distances_list.append(distances[from_idx, to_idx])

# Create the DataFrame
df = pd.DataFrame({
    'num': nums,
    'from': froms,
    'to': tos,
    'cosine_distance': cosine_distances_list
})
import seaborn as sns 
sns.boxplot(x='num', y='cosine_distance', data=df)

### 2. how similar is each element to the first element ###
# Initialize empty lists to store values for DataFrame
iterations = []
froms = []
tos = []
cosine_distances_list = []

# Loop through the 41 iterations
for iter_idx in range(num_iterations):
    # Loop through the 20 columns to compute the pairwise cosine distance
    for col_idx in range(num_per_iteration):
        # The element we will compare from (always the first in each 20-element set)
        from_element = reshaped_embeddings[iter_idx, 0, :].reshape(1, -1)
        
        # The element we will compare to
        to_element = reshaped_embeddings[iter_idx, col_idx, :].reshape(1, -1)
        
        # Compute cosine distance
        distance = cosine_distances(from_element, to_element)
        
        # Store in lists
        iterations.append(iter_idx)
        froms.append(0)  # Always comparing from the first element in each set
        tos.append(col_idx)
        cosine_distances_list.append(distance[0][0])

# Create the DataFrame
df = pd.DataFrame({
    'cosine_distance': cosine_distances_list,
    'iteration': iterations,
    'from': froms,
    'to': tos
})

sns.boxplot(x='to', y='cosine_distance', data=df)

### 3. how similar is each element across? ###
# Initialize a 2D array to store sum of cosine distances for each pair
sum_distances = np.zeros((num_per_iteration, num_per_iteration))

# Loop through the 41 iterations to accumulate the sum of cosine distances for each pair
for iter_idx in range(num_iterations):
    distances = cosine_distances(reshaped_embeddings[iter_idx, :, :])
    sum_distances += distances

# Calculate the mean distances
mean_distances = sum_distances / num_iterations

# Create the DataFrame to store mean distances
froms = []
tos = []
mean_distances_list = []

for from_idx in range(20):
    for to_idx in range(20):
        froms.append(from_idx)
        tos.append(to_idx)
        mean_distances_list.append(mean_distances[from_idx, to_idx])

df = pd.DataFrame({
    'from': froms,
    'to': tos,
    'mean_distance': mean_distances_list
})

import matplotlib.pyplot as plt 

heatmap_data = df.pivot(index="from", 
                        columns="to", 
                        values="mean_distance")

mask = np.tril(np.ones_like(heatmap_data, dtype=bool), k=-1)

# Generate the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', cbar=True, mask=mask)
plt.title("Mean Cosine Distances (Upper Triangle)")
plt.xlabel("To")
plt.ylabel("From")
plt.show()

### PCA colored by num ###



# setup
## reproduce the submitted analysis
temperature='1.0'
inpath='../data/data_output/phillips_chatgpt'
outpath='../fig/phillips_chatgpt'
contexts = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
context_mapping={i: contexts[i] for i in range(len(contexts))}

# match files (see helper_functions.py)
files_could = match_files(os.path.join(inpath, f'*temp{temperature}_could.json'))
files_should = match_files(os.path.join(inpath, f'*temp{temperature}_should.json'))

# load files (see helper_functions)
responses_could = load_files(files_could, contexts)
responses_should = load_files(files_should, contexts)

# setup
num_ctx = len(contexts)
num_gen_individual=int(len(responses_could[0])/num_ctx) # number of generations per context
num_runs=len(responses_should) # number of times we have run the above

# sort responses (see helper_functions.py)
responses_could, contexts_could = sort_responses(responses_could,
                                                 contexts,
                                                 num_gen_individual)
responses_should, contexts_should = sort_responses(responses_should,
                                                   contexts,
                                                   num_gen_individual)

# first 500 of each should fall into categories
# get all "could" responses from Brian
could_brian = [x for x, y in zip(responses_could, contexts_could) if y == 'Brian']
should_brian = [x for x, y in zip(responses_should, contexts_should) if y == 'Brian']

# seems like "could" becomes "would" here, maybe?
could_brian # almost entiry paying missing amount 
should_brian # should is reporting discrepancy

