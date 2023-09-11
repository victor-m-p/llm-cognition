'''
2023-09-11:
PCA plots used in the manuscript.
'''

import pandas as pd 
import torch
import seaborn as sns 
import matplotlib.pyplot as plt
from helper_functions import *
from transformers import AutoTokenizer, AutoModel
from matplotlib.patches import Rectangle
pd.set_option('display.max_colwidth', None)
np.random.seed(1242)

checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# setup 
n_per_iteration = 6
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
df = df.sort_values(['id', 'condition', 'iteration', 'num'])

def equalize_conditions(df: pd.DataFrame):
    """ensure equal amount of iterations for each condition

    Args:
        df (pd.DataFrame): DataFrame with columns 'condition' ('could', 'should') and 'iteration'

    Returns:
        pd.DataFrame: DataFrame with equal amount of iterations for each condition
    """
    
    # Count unique iterations for each condition
    counts = df.groupby('condition')['iteration'].nunique()
    num_could = counts.get('could', 0)
    num_should = counts.get('should', 0)

    # Determine which condition has more iterations
    largest_condition = 'could' if num_could > num_should else 'should'

    # Calculate the difference in the number of unique iterations
    diff = abs(num_could - num_should)

    # If the number of iterations is the same for both conditions, return the original dataframe
    if diff == 0:
        return df

    # Randomly select iterations to drop from the condition with more iterations
    to_remove = np.random.choice(
        df[df['condition'] == largest_condition]['iteration'].unique(), 
        diff, 
        replace=False
    )

    # Drop the selected iterations and return the modified dataframe
    return df.drop(df[(df['condition'] == largest_condition) & (df['iteration'].isin(to_remove))].index)

# loop over all vignettes
unique_vignettes = df['id'].unique()
for vignette in unique_vignettes: 

    # filter by vignette and equalize conditions
    df_ = df[df['id']==vignette]
    df_ = equalize_conditions(df_)

    # embed and encode
    responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
    responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

    # encode responses (see helper_functions.py)
    encodings_could = encode_responses(tokenizer, responses_could)
    encodings_should = encode_responses(tokenizer, responses_should)

    # embed responses (see helper_functions.py)
    embeddings_could = embed_responses(model, encodings_could)
    embeddings_should = embed_responses(model, encodings_should)

    # combine embeddings and run PCA (see helper_functions.py)
    embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
    x_temp, y_temp = run_PCA(embeddings_temp)

    # setup
    n_per_condition = embeddings_could.shape[0]
    n_iterations = int(n_per_condition/n_per_iteration)
    condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

    # create dataframe
    df_pca = pd.DataFrame({
        'x': x_temp,
        'y': y_temp,
        'context': [vignette]*n_per_condition + [vignette]*n_per_condition,
        'condition': ['could']*n_per_condition + ['should']*n_per_condition,
        'response_clean': responses_could + responses_should,
        'num': list(range(n_per_iteration)) * n_iterations * 2})
    df_pca['iter'] = (df_pca.index // n_per_iteration) % n_iterations
    df_pca['num'] = df_pca['num'] + 1
    
    # plot by num
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.scatterplot(data=df_pca,
                    x=df_pca['x'],
                    y=df_pca['y'],
                    hue='num',
                    alpha=0.5)
    
    # trace line 
    specific_condition = 'could'
    specific_iter = 0 # np.random.choice(range(n_iterations))

    # Filter data for the path line
    df_path = df_pca[(df_pca['condition'] == specific_condition) & (df_pca['iter'] == specific_iter)]

    # Save the "responses" of the points in the path to a txt file
    with open(f"../fig/gpt4/pca_path_labels/{vignette}.txt", "w") as f:
        for response in df_path['response_clean']:
            f.write(response + "\n")

    # Add path line (in black)
    ax.plot(df_path['x'], df_path['y'], color='black', marker='o')

    # Highlight start point (in black) and annotate
    start_point = df_path[df_path['num'] == 1]
    ax.scatter(start_point['x'], start_point['y'], color='black', zorder=5)
    ax.annotate('Start', (start_point['x'].values[0], start_point['y'].values[0]), textcoords="offset points", xytext=(0,10), ha='center')
                
    # general plot settings
    plt.legend(fontsize=12)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)    
    plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.png')
    plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.pdf')
                
    # plot by condition
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.scatterplot(data=df_pca,
                    x=df_pca['x'],
                    y=df_pca['y'],
                    hue='condition',
                    alpha=0.5)

    # Add path line (in black)
    ax.plot(df_path['x'], df_path['y'], color='black', marker='o')

    # Highlight start point (in black) and annotate
    start_point = df_path[df_path['num'] == 1]
    ax.scatter(start_point['x'], start_point['y'], color='black', zorder=5)
    ax.annotate('Start', (start_point['x'].values[0], start_point['y'].values[0]), 
                textcoords="offset points", xytext=(0,10), ha='center',
                fontsize=14)
    
    # general plot settings 
    plt.legend(fontsize=12)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)    
    plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}_{n_per_iteration}_cond.png')
    plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}_{n_per_iteration}_cond.pdf')

# manually check some responses and generate 
# plots with quadrants highlighted 

### Liz ###
np.random.seed(1432)
vignette = 'Liz'
df_ = df[df['id']==vignette]
df_ = equalize_conditions(df_)

# select responses
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

# encode responses (see helper_functions.py)
encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

# combine embeddings and run PCA (see helper_functions.py)
embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
x_temp, y_temp = run_PCA(embeddings_temp)

# setup
n_per_condition = embeddings_could.shape[0]
n_iterations = int(n_per_condition/n_per_iteration)
condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

# create dataframe
df_pca = pd.DataFrame({
    'x': x_temp,
    'y': y_temp,
    'context': [vignette]*n_per_condition + [vignette]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'response_clean': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})
df_pca['iter'] = (df_pca.index // n_per_iteration) % n_iterations
df_pca['num'] = df_pca['num'] + 1

# plot by condition
fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='condition',
                alpha=0.5)
# trace line 
specific_condition = 'could'
specific_iter = 0 # np.random.choice(range(n_iterations))

# Filter data for the path line
df_path = df_pca[(df_pca['condition'] == specific_condition) & (df_pca['iter'] == specific_iter)]

# Save the "responses" of the points in the path to a txt file
with open(f"../fig/gpt4/pca_path_labels/{vignette}.txt", "w") as f:
    for response in df_path['response_clean']:
        f.write(response + "\n")

# Add path line (in black)
ax.plot(df_path['x'], df_path['y'], color='black', marker='o')

# Highlight start point (in black) and annotate
start_point = df_path[df_path['num'] == 1]
ax.scatter(start_point['x'], start_point['y'], color='black', zorder=5)
ax.annotate('Start', (start_point['x'].values[0], start_point['y'].values[0]), 
            textcoords="offset points", xytext=(0,10), ha='center',
            fontsize=14)

# specific quadrants
left_ymin, left_ymax, left_xmin, left_xmax = 0.1, 0.6, -0.65, -0.25
left_ymax_minus_ymin, left_xmax_minus_xmin = left_ymax-left_ymin, left_xmax-left_xmin

right_ymin, right_ymax, right_xmin, right_xmax = -0.05, 0.4, 0.3, 0.65
right_ymax_minus_ymin, right_xmax_minus_xmin = right_ymax-right_ymin, right_xmax-right_xmin

# quick function to determine which quadrant a point is in
def determine_cluster(row):
    if (left_ymin < row['y'] < left_ymax) and (left_xmin < row['x'] < left_xmax):
        return 'left'
    elif (right_ymin < row['y'] < right_ymax) and (right_xmin < row['x'] < right_xmax):
        return 'right'
    else:
        return 'other'

# add rectangles
df_pca['cluster'] = df_pca.apply(determine_cluster, axis=1)
left_rect = Rectangle((left_xmin, left_ymin), 
                      left_xmax_minus_xmin, 
                      left_ymax_minus_ymin, 
                      color='grey', alpha=0.3)

right_rect = Rectangle((right_xmin, right_ymin),
                          right_xmax_minus_xmin,
                            right_ymax_minus_ymin,
                            color='grey', alpha=0.3)

ax.add_patch(left_rect)
ax.add_patch(right_rect)

# Calculate the percentage of "could" for each cluster
grouped = df_pca.groupby(['cluster', 'condition']).size().unstack(fill_value=0)
grouped['could_percentage'] = (grouped['could'] / (grouped['could'] + grouped['should'])) * 100

# Annotate the percentages
left_x_center = (left_xmin + (-0.23)) / 2
left_y_annotate = left_ymax-0.05  # Adjust as needed for your aesthetics
ax.annotate(f'{grouped.loc["left", "could_percentage"]:.1f}% could',
            (left_x_center, left_y_annotate), 
            ha='center', color='black',
            fontsize=14)

right_x_center = (right_xmin + (0.65)) / 2
right_y_annotate = right_ymax-0.05  # Adjust as needed for your aesthetics
ax.annotate(f'{grouped.loc["right", "could_percentage"]:.1f}% could',
            (right_x_center, right_y_annotate),
            ha='center', color='black',
            fontsize=14)

# general plot settings
plt.legend(fontsize=12)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)    
plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}_{n_per_iteration}_cond.png')
plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}_{n_per_iteration}_cond.pdf')

# manually check some responses
df_left = df_pca[df_pca['cluster']=='left']
df_left.head(10) # could
# decide to exercise outside instead, like going for a run or doing a bodyweight workout in a park
# go back to their office and use any fitness facilities there, if they are available
df_right = df_pca[df_pca['cluster']=='right']
df_right.head(10) # should
# see if there are any promotion codes or deals available that they could use to renew their membership at a discounted rate
# renew their membership on the spot if they has the money and time to do so. 

### MARY ###
# setup
np.random.seed(1412)
vignette = 'Mary'
df_ = df[df['id']==vignette]
df_ = equalize_conditions(df_)

# select responses 
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

# encode responses (see helper_functions.py)
encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

# combine embeddings and run PCA (see helper_functions.py)
embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
x_temp, y_temp = run_PCA(embeddings_temp)

# setup
n_per_condition = embeddings_could.shape[0]
n_iterations = int(n_per_condition/n_per_iteration)
condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

# create dataframe
df_pca = pd.DataFrame({
    'x': x_temp,
    'y': y_temp,
    'context': [vignette]*n_per_condition + [vignette]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'response_clean': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})
df_pca['iter'] = (df_pca.index // n_per_iteration) % n_iterations
df_pca['num'] = df_pca['num'] + 1

# plot by num
fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='num',
                alpha=0.5)
# trace line 
specific_condition = 'could'
specific_iter = 0 #np.random.choice(range(n_iterations))

# Filter data for the path line
df_path = df_pca[(df_pca['condition'] == specific_condition) & (df_pca['iter'] == specific_iter)]

# Save the "responses" of the points in the path to a txt file
with open(f"../fig/gpt4/pca_path_labels/{vignette}.txt", "w") as f:
    for response in df_path['response_clean']:
        f.write(response + "\n")

# Add path line (in black)
ax.plot(df_path['x'], df_path['y'], color='black', marker='o')

# Highlight start point (in black) and annotate
start_point = df_path[df_path['num'] == 1]
ax.scatter(start_point['x'], start_point['y'], color='black', zorder=5)
ax.annotate('Start', (start_point['x'].values[0], start_point['y'].values[0]), 
            textcoords="offset points", xytext=(0,10), ha='center',
            fontsize=14)

# specific quadrants
left_ymin, left_ymax, left_xmin, left_xmax = -0.4, 0.0, -0.4, -0.1
left_ymax_minus_ymin, left_xmax_minus_xmin = left_ymax-left_ymin, left_xmax-left_xmin

right_ymin, right_ymax, right_xmin, right_xmax = -0.45, -0.15, 0.1, 0.5
right_ymax_minus_ymin, right_xmax_minus_xmin = right_ymax-right_ymin, right_xmax-right_xmin

top_ymin, top_ymax, top_xmin, top_xmax = -0.05, 0.35, 0.25, 0.6
top_ymax_minus_ymin, top_xmax_minus_xmin = top_ymax-top_ymin, top_xmax-top_xmin

# quick function to determine which quadrant a point is in
def determine_cluster(row):
    if (left_ymin < row['y'] < left_ymax) and (left_xmin < row['x'] < left_xmax):
        return 'left'
    elif (right_ymin < row['y'] < right_ymax) and (right_xmin < row['x'] < right_xmax):
        return 'right'
    elif (top_ymin < row['y'] < top_ymax) and (top_xmin < row['x'] < top_xmax):
        return 'top'
    else:
        return 'other'

# add rectangles
df_pca['cluster'] = df_pca.apply(determine_cluster, axis=1)
left_rect = Rectangle((left_xmin, left_ymin), 
                      left_xmax_minus_xmin, 
                      left_ymax_minus_ymin, 
                      color='grey', alpha=0.3)

right_rect = Rectangle((right_xmin, right_ymin),
                       right_xmax_minus_xmin,
                       right_ymax_minus_ymin,
                       color='grey', alpha=0.3)

top_rect = Rectangle((top_xmin, top_ymin),
                     top_xmax_minus_xmin,
                     top_ymax_minus_ymin,
                     color='grey', alpha=0.3)

ax.add_patch(left_rect)
ax.add_patch(right_rect)
ax.add_patch(top_rect)

# Calculate the mean generation number by cluster and annotate
cluster_means = df_pca.groupby('cluster')['num'].mean()
left_x_center = (left_xmin + (-0.1)) / 2
left_y_annotate = left_ymax-0.35  

ax.annotate(f'Mean: {cluster_means["left"]:.1f}',
            (left_x_center, left_y_annotate), 
            ha='center', color='black', fontsize=14) 

right_x_center = (right_xmin + (0.5)) / 2
right_y_annotate = right_ymax-0.05  # Adjust as needed for your aesthetics
ax.annotate(f'Mean: {cluster_means["right"]:.1f}',
            (right_x_center, right_y_annotate), 
            ha='center', color='black', fontsize=14) 

top_x_center = (top_xmin + (0.6)) / 2
top_y_annotate = top_ymax-0.05  # Adjust as needed for your aesthetics
ax.annotate(f'Mean: {cluster_means["top"]:.1f}',
            (top_x_center, top_y_annotate),
            ha='center', color='black', fontsize=14)

# general plot settings
plt.legend(fontsize=12)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)    
plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.png')
plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.pdf')

# check what the responses are;
df_left = df_pca[df_pca['cluster']=='left']
df_left.head(10) 
# ask classmate (e.g., to borrow) or to copy "if the class rules allow for this"
# check whether they have a digital copy.
df_right = df_pca[df_pca['cluster']=='right']
df_right.head(10)
# call or text their mother (almost all exactly the same)
df_top = df_pca[df_pca['cluster']=='top']
df_top.head(10)
# explain situation to teacher (or administrator, or counsellor)

