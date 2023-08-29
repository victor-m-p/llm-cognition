'''
used in the paper
'''

import pandas as pd 
import torch
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
from helper_functions import *
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import root_scalar

checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# setup
## reproduce the submitted analysis
temperature='1.0'
inpath='../data/data_output/phillips_gpt3'
outpath='../fig/phillips_gpt3'
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

# encode sentences (see helper_functions.py)
encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

# specifically select Brad
context_dictionary={'Heinz': 0,
                    'Josh': 1,
                    'Brian': 2,
                    'Liz': 3,
                    'Mary': 4,
                    'Brad': 5}
selected_context = 5
lower=num_gen_individual*num_runs*selected_context
upper=num_gen_individual*num_runs*(selected_context+1)

# run PCA (see helper_functions.py)
embeddings_could_selected = embeddings_could[lower:upper]
embeddings_should_selected = embeddings_should[lower:upper]
embeddings_selected = torch.cat((embeddings_could_selected, embeddings_should_selected), 0)
x, y = run_PCA(embeddings_selected)
contexts_could_selected = contexts_could[lower:upper] 
contexts_should_selected = contexts_should[lower:upper]
responses_could_selected = responses_could[lower:upper]
responses_should_selected = responses_should[lower:upper]
df_pca_selected = pd.DataFrame({
    'x': x,
    'y': y,
    'context': contexts_could_selected + contexts_should_selected,
    'condition': ['could']*num_gen_individual*num_runs + ['should']*num_gen_individual*num_runs,
    'responses': responses_could_selected + responses_should_selected})

# trying to plot stuff
min_x=-1
max_x=1
line_one_intercept = 0.3
line_two_intercept = 0.05
line_two_slope = 1.1

# Function to find intersection of two lines
def find_intersection():
    f = lambda x : -x - line_one_intercept - (line_two_slope*x + line_two_intercept)
    result = root_scalar(f, bracket=[-1, 1], method='brentq')
    x_intersect = result.root
    y_intersect = -x_intersect - line_one_intercept
    return x_intersect, y_intersect

# Calculate intersection
x_intersect, y_intersect = find_intersection()

# Define x values and calculate y values based on x
x_values = np.linspace(min_x, max_x, 100)
y_red = -x_values - line_one_intercept
y_green = line_two_slope*x_values + line_two_intercept

# Values for the green line starting from the intersection
x_green_from_intersect = np.linspace(x_intersect, max_x, 50)
y_green_from_intersect = line_two_slope*x_green_from_intersect + line_two_intercept

# Create the plot
plt.figure(figsize=(8, 6))

# Scatter
sns.scatterplot(data=df_pca_selected,
                x=df_pca_selected['x'],
                y=df_pca_selected['y'],
                hue='condition',
                alpha=0.5)

# Plot red line
plt.plot(x_values, y_red, 'black', label='Red Line')

# Plot green line starting from intersection
plt.plot(x_green_from_intersect, y_green_from_intersect, 'black', label='Green Line')

# Fill regions
plt.fill_between(x_values, y_red, -1, color='tab:red', alpha=0.3)#, label='Below Red Line')
plt.fill_between(x_values, y_red, y_green, where=(y_red < y_green), color='tab:purple', alpha=0.3)#, label='Between Lines')
plt.fill_between(x_values, y_red, 1, color='tab:blue', alpha=0.3)#, label='Above Both Lines')

plt.ylim(-0.55, 0.7)
plt.xlim(-0.45, 0.8)
plt.show();


# I have sanity checked this.
def assign_group(row):
    x, y = row['x'], row['y']
    
    # Calculate y-values for the lines at this x
    y_red = -x - line_one_intercept
    y_green = line_two_slope * x + line_two_intercept
    
    # Check which region the point falls into
    if y < y_red:
        return 'tab:red'
    elif y > y_green:
        return 'tab:blue'
    else: 
        return 'tab:purple'

# Apply the function to each row in the DataFrame
df_pca_selected['group'] = df_pca_selected.apply(assign_group, axis=1)
df_pca_selected.groupby('group')['condition'].mean()
df_pca_selected.groupby('group')['condition'].value_counts()

# fractions in each (of course significant...)
df_pca_selected.groupby('group')['condition'].value_counts(normalize=True)

# qualitative
pd.set_option('display.max_colwidth', None)
df_pca_selected[df_pca_selected['group']=='tab:blue'].head(20)
df_pca_selected[df_pca_selected['group']=='tab:purple'].head(20)
df_pca_selected[df_pca_selected['group']=='tab:red'].head(20)

'''
blue (should): -- pretty varied
- contact help, find water, navigate to safety, stay in place

purple (could): -- almost all fire (if not all)
- make fire (for signal mostly, but also warmth).

red (should): -- almost all related to shelter
- create shelter, find shelter, find place to camp, ...
'''

