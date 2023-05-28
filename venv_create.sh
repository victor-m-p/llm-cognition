#!/usr/bin/env bash

VENVNAME=cppenv

python3.10 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip install --upgrade pip

CMAKE_ARGS="DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --verbose

pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

#test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
