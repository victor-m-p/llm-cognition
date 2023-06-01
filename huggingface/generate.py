'''
efficiently generate text from a given prompt. 
with gpt2 this does not work well unfortunately. 
the model is just not good enough to generate 
something coherent. at least not for a prompt of
this length. 
'''

from transformers import pipeline, set_seed
checkpoint = 'decapoda-research/llama-7b-hf' # what we want to try 
checkpoint = 'gpt2'
set_seed(42)
import xformers # just needs to be somewhere for pipreqs to understand that it should generate it in the requirements.txt
generator = pipeline('text-generation', model=checkpoint)
prompt = 'Sam is 30 years old, and has a close friend, Harry, who is struggling in life. Harry has problems with addiction, and often feels lonely and unappreciated. Harry wants to get better, but is not doing as well as he could be given his innate abilities. Sam sets a goal of helping Harry become a happier and more thriving person. The first thing Sam could do is to'
out = generator(prompt, max_length=500, num_return_sequences=5)