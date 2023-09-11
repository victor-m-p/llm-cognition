'''
VMP 2023-09-11:
This script is used to clean the data from GPT-4 (generated in generate_gpt4.py).
We gender neutralize the responses; and we create a column with shuffled responses
which is used for evaluation (evaluate_gpt4.py). 
'''
import json
import pandas as pd 
import re 
import numpy as np 
import glob

# match the files
inpath = '../data/data_output/gpt4_new/gpt-4_n100*.json'
filenames = glob.glob(inpath)

def filter_func(x, num_per_iteration=6):
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

## first we shuffle ##
def shuffle_group(group):
    group['shuffled'] = np.random.permutation(group['num'].values)
    return group

# Apply the shuffle_group function to each group
df = df.groupby(['condition', 'id', 'iteration']).apply(shuffle_group)
df = df.reset_index(drop=True)

## then we mask for responses ##

# Regular expression to capture text after 'could do is' or 'should do is'
pattern = r"(?:could do is|should do is)\s*(.*)"

# Extracting text after the pattern into a new column 'response_option'
df['response_option'] = df['response'].str.extract(pattern)

# Remove rows where 'response_option' is NaN (i.e., pattern not found)
df.dropna(subset=['response_option'], inplace=True)

# Filter again
df = df.groupby(['condition', 'id', 'iteration']).filter(filter_func)
df['response'] = df['response'].str.strip()
df['response_option'] = df['response_option'].str.strip()

# gender neutralize
#df['response_option'] = df['response_option'].str.replace('he or she', 'they')
gender_neutral_map = {
    r"\bhe\b": "they",
    r"\bhim\b": "them",
    r"\bhis\b": "their",
    r"\bshe\b": "they",
    r"\bher\b": "their"
}

# Function to replace pronouns in a case-insensitive manner
def replace_pronouns(text):
    for k, v in gender_neutral_map.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)
    return text

# Create a new column with gender-neutralized text and save
df['response_clean'] = df['response_option'].apply(replace_pronouns)
df.to_csv('../data/data_cleaned/gpt4_shuffled.csv', index=False)