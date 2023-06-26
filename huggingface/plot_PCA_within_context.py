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

# setup
temperature='1.0'
inpath='../data/data_output/phillips_gpt3'
outpath='../fig/phillips_gpt3'

# match files (see helper_functions.py)
files_could = match_files(os.path.join(inpath, f'*temp{temperature}_could_fix.json'))
files_should = match_files(os.path.join(inpath, f'*temp{temperature}_should_fix.json'))

# load files (see helper_functions)
contexts = ['Heinz', 'Josh', 'Brian', 'Liz', 'Mary', 'Brad']
responses_could = load_files(files_could, contexts)
responses_should = load_files(files_should, contexts)

# function to select dissimilar points (for labeling)
def select_dissimilar_points(df, n_points=5):
    # Function to select n points that are furthest apart in terms of Euclidean distance
    selected_indices = []
    mean_point = [np.mean(df['x']), np.mean(df['y'])]
    
    # Select the center point (closest to the mean)
    distances = np.sqrt((df['x'] - mean_point[0])**2 + (df['y'] - mean_point[1])**2)
    selected_indices.append(distances.idxmin())
    
    # Select the remaining points
    for _ in range(1, n_points):
        max_dist = 0
        max_index = None
        for index, row in df.iterrows():
            if index not in selected_indices:
                min_dist = min(np.sqrt((row['x'] - df['x'].loc[selected_indices])**2 + (row['y'] - df['y'].loc[selected_indices])**2))
                if min_dist > max_dist:
                    max_dist = min_dist
                    max_index = index
        selected_indices.append(max_index)
    
    return selected_indices

# plot PCA across conditions (within context)
def plot_PCA_across_condition(df, temperature, context_num, context, fontsize,  outpath, labels='true', use_hull='true'):
    
    # Map each unique context to a unique integer
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.scatterplot(data=df,
                    x=df['x'],
                    y=df['y'],
                    hue='condition',
                    alpha=0.5)
    
    # Get the color and style based on the group
    color = sns.color_palette()[context_num]

    # if hull is true we draw the hull
    if use_hull == 'true': 
        unique_contexts = df['condition'].unique()
        context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))
        # Generate ConvexHull for each group
        hull = ConvexHull(df[['x', 'y']])
        for (cond, ), group in df.groupby(['condition']):
            # Generate ConvexHull for each group
            hull = ConvexHull(group[['x', 'y']])
            
            # Get the color and style based on the group
            color = sns.color_palette()[context_to_int[cond]]
            #style = '-' if cond == 'could' else '--'  # adjust according to your conditions
            
            # Draw the lines and fill the areas
            for simplex in hull.simplices:
                plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], color=color) #style, color=color)
            plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)

    # Label 5 random points
    if labels=='true': 
        #selected_indices = random.sample(range(len(df_context)), 5)
        selected_indices = select_dissimilar_points(df, 6)
        responses = []
        total_lines = 0
        for i, index in enumerate(selected_indices, 1):
            x = df['x'].loc[index]
            y = df['y'].loc[index]
            response = df['responses'].loc[index]
            # Insert line breaks for long responses
            max_line_length = 75
            response_lines = textwrap.wrap(response, width=max_line_length)
            #response_lines = [response[j:j+max_line_length] for j in range(0, len(response), max_line_length)]
            responses.append(f"{i}: " + "\n".join(response_lines))
            ax.text(x, y, str(i), fontsize=fontsize, ha='right')

            # Dynamically calculate vertical position
            for line in response_lines:
                plt.figtext(0.1, -0.05 - 0.05 * total_lines, line if line != response_lines[0] else f"{i}: {line}", 
                            fontsize=12, ha="left")
                total_lines += 1
    
    plt.legend(fontsize=fontsize)
    plt.xlabel('PC1', fontsize=fontsize)
    plt.ylabel('PC2', fontsize=fontsize)    
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, f'fig_png/PCA_hull_{use_hull}_temp{temperature}_{context}_labels_{labels}_fix.png'), bbox_inches='tight')
    fig.savefig(os.path.join(outpath, f'fig_pdf/PCA_hull_{use_hull}_temp{temperature}_{context}_labels_{labels}_fix.pdf'))
    plt.close()

context_mapping={0: 'Heinz',
                 1: 'Josh',
                 2: 'Brian',
                 3: 'Liz',
                 4: 'Mary',
                 5: 'Brad'}

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
encodings_could = encode_responses(responses_could)
encodings_should = encode_responses(responses_should)

# embed responses (see helper_functions.py)
embeddings_could = embed_responses(encodings_could)
embeddings_should = embed_responses(encodings_should)

for num, i in enumerate(range(0, num_ctx*num_gen_individual*num_runs, num_gen_individual*num_runs)):
    # embeddings for could and should
    embeddings_could_temp = embeddings_could[i:i+num_gen_individual*num_runs]
    embeddings_should_temp = embeddings_should[i:i+num_gen_individual*num_runs]
    # stack and run pca 
    embeddings_temp = torch.cat((embeddings_could_temp, embeddings_should_temp), 0)
    x_temp, y_temp = run_PCA(embeddings_temp)
    # contexts for could and should
    contexts_could_temp = contexts_could[i:i+num_gen_individual*num_runs]
    contexts_should_temp = contexts_should[i:i+num_gen_individual*num_runs]
    # responses for could and should
    responses_could_temp = responses_could[i:i+num_gen_individual*num_runs]
    responses_should_temp = responses_should[i:i+num_gen_individual*num_runs]
    # gather in dataframe
    df_pca_temp = pd.DataFrame({
        'x': x_temp,
        'y': y_temp,
        'context': contexts_could_temp + contexts_should_temp,
        'condition': ['could']*num_gen_individual*num_runs + ['should']*num_gen_individual*num_runs,
        'responses': responses_could_temp + responses_should_temp})
    # context mapping
    context_temp = context_mapping.get(num)
    # plot it 
    plot_PCA_across_condition(df_pca_temp,
                              temperature,
                              num,
                              context_temp,
                              18,
                              outpath,
                              'true',
                              'false')
