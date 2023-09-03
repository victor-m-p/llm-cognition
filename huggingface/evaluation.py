import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

eval = pd.read_csv('../data/data_output/gpt4_eval/results.csv')
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
df = df.merge(eval, on=['condition', 'id', 'iteration', 'shuffled'], how='inner')
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