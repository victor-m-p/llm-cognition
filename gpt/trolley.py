import openai 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import requests 
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = 'A trolley is heading towards 2 people. You can either choose option 1 and pull the lever which will divert the trolley to kill 1 person, or choose option 2 and do nothing which will kill 2 people. You should choose option number'
prompt = 'A trolley is heading towards 2 people. You can either choose option 1 and do nothing which will kill 2 people, or choose option 2 and pull the lever which will divert the trolley to kill 1 person. You should choose option number'

# this gives actually good results
openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  max_tokens=10, # for completion (not including prompt)
  temperature=1, # (0, 1)
  logprobs=5 # the maximum 
) # utilitarian