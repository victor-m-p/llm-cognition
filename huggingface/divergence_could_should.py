import pandas as pd 
import numpy as np 
from helper_functions import *

df = pd.read_csv('../data/data_cleaned/gpt4_eval.csv')

# set this up in a good way...
# basically compare: 
# compare all (1, 1) for each (condition, id) across iter.

could = df[(df['condition'] == 'could') & (df['id'] == 'Mary')]
should = df[(df['condition'] == 'should') & (df['id'] == 'Mary')]


# Step 1: Group the DataFrame by 'id', 'condition', 'iter', 'num'
grouped = df.groupby(['id', 'condition', 'iter', 'num'])

# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['id', 'num', 'cosine_distance'])

# Step 2: Iterate through the groups
for (id_val, condition_val, iter_val, num_val), group_data in grouped:
    if len(group_data) == 0:
        continue
    # Collect the 'response_clean' values and encode them
    encoded_responses = group_data['response_clean'].apply(encode).tolist()
    
    # Now compare this list across conditions and iterations for each 'num' and 'id'
    for other_condition_val, other_group_data in grouped:
        if other_condition_val[0] != id_val or other_condition_val[1] == condition_val or other_condition_val[3] != num_val:
            continue
        other_encoded_responses = other_group_data['response_clean'].apply(encode).tolist()
        
        # Calculate the cosine distance between each pair of 'response_clean' across conditions
        for resp in encoded_responses:
            for other_resp in other_encoded_responses:
                distance = cosine_distance(resp, other_resp)
                result_df = result_df.append({
                    'id': id_val,
                    'num': num_val,
                    'cosine_distance': distance
                }, ignore_index=True)