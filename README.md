# llm-cognition
Code for "Large Language Models in the Labyrinth: Possibility Spaces and Moral Constraints" submitted 23-06-2023 to Possibility Studies and Society.

Tested on POP!\_OS 22.04 LTS with NVIDIA GA107M (GeForce RTX 3050 Ti Mobile) graphics card.

# Overview
## Data Generation
Data generated in ```gpt/generate_gpt4.py```
Preprocessed in ```gpt/preprocessing.py```
Evaluation in ```gpt/evaluate_gpt4.py```

## Results Generation
PCA plots in ```huggingface/plot_PCA.py```
Raincloud pplot in ```huggingface/plot_raincloud.py```
Rank order correlation in ```huggingface/rank_order_corr.py```
Evaluation of generation number ```huggingface/generation_number.py```

# Set up environments

## /gpt env (creating data, requires API key)
```bash
cd gpt
bash create_venv.sh
source gptenv/bin/activate
```

## /huggingface env (creating plots)
```bash
cd huggingface
bash create_venv.sh
source hugenv/bin/activate
```

