'''
VMP 2023-09-11:
This script is used to check a claim made in the discussion.
'''

import pandas as pd 

# load data
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')
df_heinz = df[df['id'] == 'Heinz']

# check for words
words=['steal', 'stole', 'rob', 'thie']
pattern = '|'.join(words)
word_match = df_heinz[df_heinz['response_option'].str.contains(pattern, case=False, na=False)]