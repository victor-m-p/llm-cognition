'''
used in the paper
'''

import pandas as pd 
import torch
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
from helper_functions import *
from scipy.spatial import ConvexHull
import textwrap 
from transformers import AutoTokenizer, AutoModel
from matplotlib.patches import Rectangle
pd.set_option('display.max_colwidth', None)
np.random.seed(1242)

checkpoint = 'sentence-transformers/all-MiniLM-L12-v2' 
model = AutoModel.from_pretrained(checkpoint) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# setup 
# 1. gpt4.csv and n_per_iteration=20
# 2. gpt4_subset.csv and n_per_iteration=18 (removeing first and last option)
n_per_iteration = 6
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
df = df.sort_values(['id', 'condition', 'iteration', 'num'])

# function to select dissimilar points (for labeling)
'''
def label_points(df): 
    selected_indices = np.random.choice(df.index, 6, replace=False)
    responses = []
    total_lines = 0
    for i, index in enumerate(selected_indices, 1):
        x = df['x'].loc[index]
        y = df['y'].loc[index]
        response = df['response_clean'].loc[index]
        # Insert line breaks for long responses
        max_line_length = 75
        response_lines = textwrap.wrap(response, width=max_line_length)
        #response_lines = [response[j:j+max_line_length] for j in range(0, len(response), max_line_length)]
        responses.append(f"{i}: " + "\n".join(response_lines))
        ax.text(x, y, str(i), fontsize=12, ha='right')

        # Dynamically calculate vertical position
        for line in response_lines:
            plt.figtext(0.1, -0.05 - 0.05 * total_lines, line if line != response_lines[0] else f"{i}: {line}", 
                        fontsize=12, ha="left")
            total_lines += 1
'''

# double check this code 
def equalize_conditions(df):
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

    df_ = df[df['id']==vignette]
    df_ = equalize_conditions(df_)
    
    # because we are plotting could and should together, we should anonymize
    # otherwise we could be inducing a "false" effect

    # embed and encode
    responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
    responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

    encodings_could = encode_responses(tokenizer, responses_could)
    encodings_should = encode_responses(tokenizer, responses_should)

    embeddings_could = embed_responses(model, encodings_could)
    embeddings_should = embed_responses(model, encodings_should)

    embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
    x_temp, y_temp = run_PCA(embeddings_temp)

    n_per_condition = embeddings_could.shape[0]
    n_iterations = int(n_per_condition/n_per_iteration)

    condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

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
    ax.annotate('Start', (start_point['x'].values[0], start_point['y'].values[0]), textcoords="offset points", xytext=(0,10), ha='center')
                
    #label_points(df_pca)
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
      
    plt.legend(fontsize=12)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)    
    plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}_{n_per_iteration}_cond.png')
    plt.savefig(f'../fig/gpt4/pca_could_should/{vignette}_{n_per_iteration}_cond.pdf')

### Liz ###
np.random.seed(1432)
vignette = 'Liz'
df_ = df[df['id']==vignette]
df_ = equalize_conditions(df_)

# embed and encode
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
x_temp, y_temp = run_PCA(embeddings_temp)

n_per_condition = embeddings_could.shape[0]
n_iterations = int(n_per_condition/n_per_iteration)

condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

df_pca = pd.DataFrame({
    'x': x_temp,
    'y': y_temp,
    'context': [vignette]*n_per_condition + [vignette]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'response_clean': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})
df_pca['iter'] = (df_pca.index // n_per_iteration) % n_iterations
df_pca['num'] = df_pca['num'] + 1

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.scatterplot(data=df_pca,
                x=df_pca['x'],
                y=df_pca['y'],
                hue='condition',
                alpha=0.5)
# trace line 
specific_condition = 'could'
specific_iter = 0 #np.random.choice(range(n_iterations))

# Filter data for the path line
df_path = df_pca[(df_pca['condition'] == specific_condition) & (df_pca['iter'] == specific_iter)]

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
def determine_cluster(row):
    if (left_ymin < row['y'] < left_ymax) and (left_xmin < row['x'] < left_xmax):
        return 'left'
    # You can add conditions for 'mid_bottom' similarly:
    elif (right_ymin < row['y'] < right_ymax) and (right_xmin < row['x'] < right_xmax):
        return 'right'
    else:
        return 'other'

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

grouped = df_pca.groupby(['cluster', 'condition']).size().unstack(fill_value=0)

# Calculate the percentage of "could" for each cluster
grouped['could_percentage'] = (grouped['could'] / (grouped['could'] + grouped['should'])) * 100

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

'''
### BRAD ###
np.random.seed(1412)
vignette = 'Brad'
df_ = df[df['id']==vignette]
df_ = equalize_conditions(df_)

# embed and encode
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
x_temp, y_temp = run_PCA(embeddings_temp)

n_per_condition = embeddings_could.shape[0]
n_iterations = int(n_per_condition/n_per_iteration)

condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

df_pca = pd.DataFrame({
    'x': x_temp,
    'y': y_temp,
    'context': [vignette]*n_per_condition + [vignette]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'response_clean': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})
df_pca['iter'] = (df_pca.index // n_per_iteration) % n_iterations
df_pca['num'] = df_pca['num'] + 1

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
left_ymin, left_ymax, left_xmin, left_xmax = -0.1, 0.3, -0.6, -0.3
left_ymax_minus_ymin, left_xmax_minus_xmin = left_ymax-left_ymin, left_xmax-left_xmin

right_ymin, right_ymax, right_xmin, right_xmax = -0.45, 0.6, -0.25, 0.6
right_ymax_minus_ymin, right_xmax_minus_xmin = right_ymax-right_ymin, right_xmax-right_xmin
def determine_cluster(row):
    if (left_ymin < row['y'] < left_ymax) and (left_xmin < row['x'] < left_xmax):
        return 'left'
    # You can add conditions for 'mid_bottom' similarly:
    elif (right_ymin < row['y'] < right_ymax) and (right_xmin < row['x'] < right_xmax):
        return 'right'
    else:
        return 'other'

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

cluster_means = df_pca.groupby('cluster')['num'].mean()

left_x_center = (left_xmin + (-0.3)) / 2
left_y_annotate = left_ymax-0.05  # Adjust as needed for your aesthetics

ax.annotate(f'Mean: {cluster_means["left"]:.1f}',
            (left_x_center, left_y_annotate), 
            ha='center', color='black', fontsize=14) 

right_x_center = (right_xmin + (1.1)) / 2
right_y_annotate = right_ymax-0.1  # Adjust as needed for your aesthetics
ax.annotate(f'Mean: {cluster_means["right"]:.1f}',
            (right_x_center, right_y_annotate), 
            ha='center', color='black', fontsize=14) 

plt.legend(fontsize=12)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)    
plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.png')
plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.pdf')

# manually check some responses
df_left = df_pca[df_pca['cluster']=='left']
df_left.head(10) # could
# ration (conserve) food, supplies, water, energy 
df_right = df_pca[df_pca['cluster']=='right']
df_right.head(10) # should
# create distress signal, start a fire, build a shelter
# look for water and food, keep morale high, and stay positive.
'''


#### MARY ####
np.random.seed(1412)
vignette = 'Mary'
df_ = df[df['id']==vignette]
df_ = equalize_conditions(df_)

# embed and encode
responses_could = df_[df_['condition']=='could']['response_clean'].tolist()
responses_should = df_[df_['condition']=='should']['response_clean'].tolist()

encodings_could = encode_responses(tokenizer, responses_could)
encodings_should = encode_responses(tokenizer, responses_should)

embeddings_could = embed_responses(model, encodings_could)
embeddings_should = embed_responses(model, encodings_should)

embeddings_temp = torch.cat((embeddings_could, embeddings_should), 0)
x_temp, y_temp = run_PCA(embeddings_temp)

n_per_condition = embeddings_could.shape[0]
n_iterations = int(n_per_condition/n_per_iteration)

condition_order = ['could']*int((n_per_condition/2)) + ['should']*int((n_per_condition/2))

df_pca = pd.DataFrame({
    'x': x_temp,
    'y': y_temp,
    'context': [vignette]*n_per_condition + [vignette]*n_per_condition,
    'condition': ['could']*n_per_condition + ['should']*n_per_condition,
    'response_clean': responses_could + responses_should,
    'num': list(range(n_per_iteration)) * n_iterations * 2})
df_pca['iter'] = (df_pca.index // n_per_iteration) % n_iterations
df_pca['num'] = df_pca['num'] + 1

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

def determine_cluster(row):
    if (left_ymin < row['y'] < left_ymax) and (left_xmin < row['x'] < left_xmax):
        return 'left'
    # You can add conditions for 'mid_bottom' similarly:
    elif (right_ymin < row['y'] < right_ymax) and (right_xmin < row['x'] < right_xmax):
        return 'right'
    elif (top_ymin < row['y'] < top_ymax) and (top_xmin < row['x'] < top_xmax):
        return 'top'
    else:
        return 'other'

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

cluster_means = df_pca.groupby('cluster')['num'].mean()

left_x_center = (left_xmin + (-0.1)) / 2
left_y_annotate = left_ymax-0.35  # Adjust as needed for your aesthetics

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

plt.legend(fontsize=12)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)    
plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.png')
plt.savefig(f'../fig/gpt4/pca_num/{vignette}_{n_per_iteration}_num.pdf')