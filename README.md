# llm-cognition
Tested on POP!\_OS 22.04 LTS with NVIDIA GA107M (GeForce RTX 3050 Ti Mobile) graphics card.
Parts of the code will require modification to run on other OS and graphics. 

# Overview
Data generated in gpt/genereate\_branch\_completion.py
Plots generated from huggingface/plot\_raincloud.py, huggingface/plot\_PCA\_within\_context.py.

# Creating environments

## HuggingFace env
```bash
cd huggingface
bash create_venv.sh
source hugenv/bin/activate
```

## OpenAI env
```bash
cd gpt
bash create_venv.sh
source gptenv/bin/activate
```
