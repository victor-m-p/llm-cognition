import numpy as np 
import pandas as pd 

df = pd.read_csv('../data/data_cleaned/gpt4_eval.csv')
df['difference']=np.abs(df['ranking']-df['num'])

df.groupby(['condition', 'id'])['difference'].mean()

# really strong for some of them
# really not strong for others ...
# do we just have insufficient data?
# I think it considers all of them pretty good...