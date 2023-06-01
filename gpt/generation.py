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

prompt = 'Sam is 30 years old, and has a close friend, Harry, who is struggling in life. Harry has problems with addiction, and often feels lonely and unappreciated. Harry wants to get better, but is not doing as well as he could be given his innate abilities. Sam sets a goal of helping Harry become a happier and more thriving person. The first thing Sam could do is to'

# this gives actually good results
openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  max_tokens=30, # for completion (not including prompt)
  temperature=1, # (0, 1)
  logprobs=5 # the maximum 
)

## very constrained possibility space here it seems (surprised that even highest temperature is so narrow).
## perhaps there are some tricky things where it is just only allowed to give very general and vague advice. 
## this would actually be super interesting; i.e. if these models are now so constrained as to be unhelpful. 
# temperature=0: listen to Harry and be a supportive friend. He can encourage Harry to talk about his feelings and provide a safe space for him to express himself. Sam
# temperature=0.5: listen to Harry and provide emotional support. He could encourage Harry to seek help from a professional, such as a therapist or addiction counselor. He could also
# temperature=1: listen to and support Harry. He can be available to talk to Harry about his issues and provide emotional and psychological support to his friend. Sam can also

## should test this against the LLaMA model that we have. 