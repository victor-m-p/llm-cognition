# llm-cognition
Code for "Large Language Models in the Labyrinth: Possibility Spaces and Moral Constraints" submitted 23-06-2023 to Possibility Studies and Society.

Tested on POP!\_OS 22.04 LTS with NVIDIA GA107M (GeForce RTX 3050 Ti Mobile) graphics card.
Parts of the code will require modification to run on other OS and graphics. 

# Overview
Data generated in ```gpt/genereate\_branch\_completion.py```
Plots generated in ```huggingface/plot\_raincloud.py```, ```huggingface/plot\_PCA\_within\_context.py.```

# Set up environments

## OpenAI env (creating data, requires API key)
```bash
cd gpt
bash create_venv.sh
source gptenv/bin/activate
```

## HuggingFace env (creating plots)
```bash
cd huggingface
bash create_venv.sh
source hugenv/bin/activate
```

