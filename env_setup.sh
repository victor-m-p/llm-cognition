#!/usr/bin/env bash

conda update -n base -c defaults conda
sudo apt install cmake
pip install ipython
pip install jupyter
conda env create -f environment.yml
conda activate llm-cognition 
#pip install scikit-build
#python -m ipykernel install --user --name=$VENVNAME

#test -f requirements.txt && pip install -r requirements.txt

#deactivate
#echo "build $VENVNAME"
