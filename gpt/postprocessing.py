import pandas as pd 
import numpy as np

prompts = pd.read_csv('../data/data_output/gpt4_eval/prompts.csv')
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')

df_list = []
for index, row in prompts.iterrows():
    generation = generation_list_outer[index]
    
    shuffled = [int(x.strip()) for x in generation.split(',')]

    # Create a sorted version of the list
    ranking = sorted(shuffled)

    # Create DataFrame
    d_ = pd.DataFrame({
        'ranking': ranking,
        'shuffled': shuffled
    })
    
    d_['condition'] = row['condition']
    d_['id'] = row['id']
    d_['iteration'] = row['iteration']
    
    df_list.append(d_)

df_prompts = pd.concat(df_list)

df_eval = df.merge(df_prompts, 
                   on=['condition', 'id', 'iteration', 'shuffled'],
                   how='inner')

# fix gender
def replace_words(text):
    text = re.sub(r'\b(she|he)\b', 'they', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(her|his)\b', 'their', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(him|her)\b', 'them', text, flags=re.IGNORECASE)
    return text

df_eval['response_clean'] = df_eval['response_option'].apply(lambda x: replace_words(x))

# sort by condition, id, iteration, num
df_eval = df_eval.sort_values(by=['condition', 'id', 'iteration', 'num'])

# save
df_eval.to_csv('../data/data_cleaned/gpt4_eval.csv', index=False)

