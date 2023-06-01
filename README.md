# llm-cognition
Tested on POP!_OS 22.04 LTS with NVIDIA GA107M (GeForce RTX 3050 Ti Mobile) graphics card.
Parts of the code will require modification to run on other OS and graphics. 

## HuggingFace pipeline
```bash
cd huggingface
bash create_venv.sh
source hugenv/bin/activate
```

now we can e.g. run the pipeline which gives probabilities and entropy of document
```bash
python probability_name_main.py -i data_input/ -o data_output 
```

## GPT pipeline
```bash
cd gpt
bash create_venv.sh
source gptenv/bin/activate
```

## 4-bit Quantized local (LLaMA) pipeline
to be documented and included.