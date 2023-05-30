# llm-cognition
Tested on POP!_OS 22.04 LTS with NVIDIA GA107M (GeForce RTX 3050 Ti Mobile) graphics card.
Parts of the code will require modification to run on other OS and graphics. 

## HuggingFace pipeline
```bash
bash venv_hugging.sh
source hugenv/bin/activate
python pipeline.py -i data_input/ -o data_output
```

## 4-bit Quantized local (LLaMA) pipeline
to be documented and included.