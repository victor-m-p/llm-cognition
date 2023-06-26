'''
Not used in the current manuscript
'''

# testing plotting of the phillips answers from the 2017 paper
# not included in the manuscript currently. 
with open('data_input/phillips2017.json') as f: 
        phillips2017 = json.load(f)

ctx = list(phillips2017.keys())
ctx_dict = {'context_1': 'Heinz',
            'context_2': 'Josh',
            'context_3': 'Brian',
            'context_4': 'Liz',
            'context_5': 'Mary',
            'context_6': 'Brad'}

ordinary=phillips2017['context_1']['ordinary']
immoral=phillips2017['context_1']['immoral']
#impossible=phillips2017['context_1']['impossible']
#improbable=phillips2017['context_1']['improbable']
#irrational=phillips2017['context_1']['irrational']

# wrangle 
ctx_idx=0
ctx=ctx_dict.get(f'context_{ctx_idx+1}')
phillips2017[f'context_{ctx_idx+1}']['immoral']

responses_could_test=responses_could[0:num_gen_individual*num_runs]
conditions_could_test=['gpt' for i in range(num_gen_individual*num_runs)]
contexts_could_test=[ctx_dict.get(f'context_{ctx_idx+1}') for i in range(num_gen_individual*num_runs)]
keys=['ordinary', 'immoral']
for i in keys:
    response_add=phillips2017[f'context_{ctx_idx+1}'][i]
    context_add=[ctx_dict.get(f'context_{ctx_idx+1}') for i in range(len(response_add))]
    conditions_add=[i for _ in range(len(response_add))]
    responses_could_test=responses_could_test+response_add
    contexts_could_test=contexts_could_test+context_add
    conditions_could_test=conditions_could_test+conditions_add
#responses_could_test=[item for sublist in responses_could_test for item in sublist]
#onditions_could_test=[item for sublist in conditions_could_test for item in sublist]
#contexts_could_test=[item for sublist in contexts_could_test for item in sublist]

# embed 
encodings_could_test = encode_responses(responses_could_test)
embeddings_could_test = embed_responses(encodings_could_test)
x_could_test, y_could_test = run_PCA(embeddings_could_test)

# construct relevant dataframe
df_pca_could_test = pd.DataFrame({
    'x': x_could_test,
    'y': y_could_test,
    'context': contexts_could_test,
    'condition': conditions_could_test,
    'responses': responses_could_test})

# plot this somehow 
def run_PCA_single_context2(df, temperature, condition, labels='true'):

    context=df['context'].unique()[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df,
                    x=df['x'],
                    y=df['y'],
                    hue='condition',
                    alpha=0.5)

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
            ax.text(x, y, str(i), fontsize=12, ha='right')

            # Dynamically calculate vertical position
            for line in response_lines:
                plt.figtext(0.1, -0.05 - 0.05 * total_lines, line if line != response_lines[0] else f"{i}: {line}", 
                            fontsize=12, ha="left")
                total_lines += 1
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.tight_layout()
    #fig.savefig(f'fig_png/PCA_phillips_temp{temperature}_{condition}_{context}_labels_{labels}_fix.png', bbox_inches='tight')
    #fig.savefig(f'fig_pdf/PCA_phillips_temp{temperature}_{condition}_{context}_labels_{labels}_fix.pdf') 
    plt.show();

run_PCA_single_context2(df_pca_could_test,
                        temperature,
                        'could',
                        'true')
                        
                        

# plot PCA across conditions
def plot_PCA_ctx_cond_hull(df, title, outname):

    # Map each unique context to a unique integer
    unique_contexts = df['context'].unique()
    context_to_int = dict(zip(unique_contexts, range(len(unique_contexts))))

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, 
                    x='x', 
                    y='y', 
                    hue='context', 
                    style='condition',
                    alpha=0.5)

    for (ctxt, cond), group in df.groupby(['context', 'condition']):
        # Generate ConvexHull for each group
        hull = ConvexHull(group[['x', 'y']])
        
        # Get the color and style based on the group
        color = sns.color_palette()[context_to_int[ctxt]]
        style = '-' if cond == 'could' else '--'  # adjust according to your conditions
        
        # Draw the lines and fill the areas
        for simplex in hull.simplices:
            plt.plot(group['x'].iloc[simplex], group['y'].iloc[simplex], style, color=color)
        plt.fill(group['x'].iloc[hull.vertices], group['y'].iloc[hull.vertices], alpha=0.5, color=color)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(f'fig_png/{outname}.png')
    fig.savefig(f'fig_pdf/{outname}.pdf')
    plt.close()
    
plot_PCA_ctx_cond_hull(df_pca_grouped, 
                       'PCA projection across conditions', 
                       f'PCA_hull_temp{temperature}_grouped')

# (add Jonathan Phillips generated possibilities) 