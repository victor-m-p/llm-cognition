import json
import pandas as pd 
import re 
import os 
import glob

inpath = '../data/data_output/gpt4/*.json'
filenames = glob.glob(inpath)

def filter_func(x, num_per_iteration=20):
    return len(x) == num_per_iteration 

df_list = []
for file in filenames:
    # Load the JSON file
    with open(file) as f:
        dct = json.load(f)
    
    # Initialize empty lists to hold DataFrame values
    condition = re.search(r'\d+_p\d+\.\d+_(.*?)\.json', file).group(1)
    ids = []
    vignettes = []
    prompts = []
    iterations = []
    nums = []
    responses = []

    # Loop through dictionary
    for context, values in dct.items():
        id_ = values['id']
        vignette = values['vignette']
        prompt = values['prompt']
        generations = values['generation']
        
        iteration = 1  # Counter for the iteration
        
        for generation in generations:
            # Split generation into individual responses based on the numbering
            individual_responses = re.split(r'\d+\.', generation)
            
            # Remove any empty strings
            individual_responses = [x.strip() for x in individual_responses if x.strip()]
            
            for num, response in enumerate(individual_responses, 1):
                ids.append(id_)
                vignettes.append(vignette)
                prompts.append(prompt)
                iterations.append(iteration)
                nums.append(num)
                responses.append(response)
            
            iteration += 1  # Increment the iteration counter

    # Create the DataFrame
    df = pd.DataFrame({
        'condition': condition,  
        'id': ids,
        'vignette': vignettes,
        'prompt': prompts,
        'iteration': iterations,
        'num': nums,
        'response': responses
    })
    
    # Only iterations that are complete
    df = df.groupby(['iteration']).filter(filter_func)
    
    # Append to list of DataFrames
    df_list.append(df)

# Concatenate all DataFrames
df = pd.concat(df_list)

# clean text
# some things are really tricky and solved heuristically
# 1. sometimes a name should be "they" and sometimes "them" (solved heuristically)
# 2. sometimes changing from he/she to they should change following verb gramatically.
# this means that there will be some errors in the data. 
def preprocess_text(text, names):
    # Gender-neutral replacements, maintaining capitalization
    gender_neutral_dict = {
        r'\b(he|He)\b': 'they',
        r'\b(she|She)\b': 'they',
        r'\b(his|His)\b': 'their',
        r'\b(her|Her)\b': 'their',
        r'\b(him|Him)\b': 'them'
    }
    
    for pattern, replacement in gender_neutral_dict.items():
        text = re.sub(pattern, lambda m: replacement if m.group().islower() else replacement.capitalize(), text)

    # Replace should with could, maintaining capitalization
    text = re.sub(r'\b(should|Should)\b', lambda m: 'could' if m.group().islower() else 'Could', text)

    # Contextual name replacement
    for name in names:
        pattern = r'([.!?]?[\s]*)\b{}\b[\s]*(\w*)'.format(re.escape(name))
        text = re.sub(pattern, lambda m: f"{m.group(1)}{'them' if m.group(2) in ['to', 'for', 'with', 'by', 'at'] else 'they'} {m.group(2)}", text)
        
    return text

vignettes = df['id'].unique()
df['response_clean'] = df['response'].apply(lambda x: preprocess_text(x, vignettes))

# create dataset where we remove the first and last response
df_subset = df[(df['num'] != 1) & (df['num'] != 20)]

df.to_csv('../data/data_cleaned/gpt4.csv', index=False)
df_subset.to_csv('../data/data_cleaned/gpt4_subset.csv', index=False)