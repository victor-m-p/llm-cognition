#!/usr/bin/env bash

VENVNAME=gptenv

python3.10 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip install --upgrade pip
pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
