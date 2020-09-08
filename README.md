# Subdirectory for python package

## Setup python and create a virtual environment for development
Install python
```
sudo apt-get install python3.7-dev
```
Setup a virtual environment named `venv_dml_dev`
```
virtualenv -p /usr/bin/python3.7 venv_dml_dev
```

## Start the virtual environment and setup package
Activate the virtual environment
```
source venv_dml_dev/bin/activate
```
Install requirements
```
pip install -r requirements.txt
```
Run `setup.py` of package `dml`
```
python doubleml/setup.py develop
```
Test whether package can be loaded
```
python doubleml/doubleml/tests/test_package.py
```

_______

## With Anaconda: Setup python and create a virtual environment for development

Setup a virtual environment named `venv_dml_dev` with python 3.7
```
conda create -n venv_dml_dev pip python=3.7
```

## Start the virtual environment and setup package
Activate the virtual environment
```
conda activate venv_dml_dev
```
Install requirements
```
pip install -r requirements.txt
```
Run `setup.py` of package `dml`
```
python doubleml/setup.py develop
```
Test whether package can be loaded
```
python doubleml/doubleml/tests/test_package.py
```
Finally, deactivate virtual environment again
```
conda deactivate venv_dml_dev
```

