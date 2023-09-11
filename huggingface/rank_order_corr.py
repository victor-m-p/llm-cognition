'''
VMP 2023-09-11:
compute spearrman rank order correlation
between the generation order, the shuffled order,
and the evaluation order (across and within conditions).

Also generates a plot which we have not 
included in the manuscript. 
'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats

# load data
eval = pd.read_csv('../data/data_output/gpt4_eval/results.csv')
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
df = df.merge(eval, on=['condition', 'id', 'iteration', 'shuffled'], how='inner')

# compute spearman rank order correlation across conditions
num_rank = stats.spearmanr(df['num'], df['eval'])
rank_shuffled = stats.spearmanr(df['shuffled'], df['eval'])
shuffled_num = stats.spearmanr(df['shuffled'], df['num'])

num_rank # r(7150) = 0.4, p < 0.001
rank_shuffled # r(7150) = 0.33, p < 0.001
shuffled_num # r(7150) = -0.01, p = 0.49

# compute spearman rank order correlation within conditions
dct_rank_order = {}
vignettes = df['id'].unique()
for vignette in vignettes:
    vignette_df = df[df['id'] == vignette]
    N = len(vignette_df)-2
    num_rank = stats.spearmanr(vignette_df['num'], vignette_df['eval'])
    rank_shuffled = stats.spearmanr(vignette_df['shuffled'], vignette_df['eval'])
    shuffled_num = stats.spearmanr(vignette_df['shuffled'], vignette_df['num'])
    dct_rank_order[vignette] = [num_rank, rank_shuffled, shuffled_num, N]

# plot evaluation (not used in paper)
df['difference'] = np.abs(df['eval'] - df['num'])
df['baseline'] = np.abs(df['shuffled'] - df['num'])

# Group by 'condition' and 'id' and compute mean of 'difference'
df_means = df.groupby(['condition', 'id'])['difference'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_means, x='id', y='difference', hue='condition', style='condition', s=100)

# Add vertical line
plt.axhline(y=1.944, color='red', linestyle='--')

plt.title('Mean of Difference for Each ID and Condition')
plt.savefig('../fig/gpt4/evaluation.png')