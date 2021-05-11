WhatsAppnalysis
----

Load and analyze WhatsApp chat history, e.g. as exported to a text file.

# Developer setup

## Prerequisites
- [`poetry`](https://python-poetry.org/)
- python 3.7 environment, e.g. using [`pyenv`](https://github.com/pyenv/pyenv)

## Setup

Create virtual env. From the project root, run:
```
whatsappnalysis$ poetry install
```

Activate the venv
```
whatsappnalysis$ source .venv/bin/activate
```


# Usage

## Setup data
Put raw chat history text file in `/data/01_raw/`. Additional pipeline datasets
will be saved in subsequent subfolders of `/data/`

## Update configuration

Pipline configuration is set in `/whatsappnalysis/config_pipeline.py`

LSTM model configuration is set in `/whatsappnalysis/config_model.py`

## Run pipeline
From within the virtual environment:
```
(.venv) whatsappnalysis$ poetry run -m python whatsappnalysis.pipeline
```

## Run notebooks

Jupyter notebooks can be used/displayed/run via Jupyter lab.
To start up jupyter lab:
```
(.venv) whatsappnalysis$ jupyter lab
```
This will open jupyter lab in a browser. From there you can
open and run the notebooks

**Note:**
Notebook outputs should be cleared before committing. To clear
notebook outputs, do one of:
* From jupyter lab: `Kernel` > `Restart kernel and clear all outputs`
* Or from a terminal: `jupyter nbconvert --clear-output --inplace my_notebook.ipynb`

## Run tests

To run tests using pytest run
```bash
pytest tests


## EC2 setup

To setup environment on new Amazon Linux 2 machine...

**Install libraries**:
```bash
sudo yum install -y \
    gcc \
    libbz2-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    tk-dev \
    liblzma-dev \
    sqlite-devel \
    libffi-devel \
    openssl-devel \
    zlib-devel \
    bzip2-devel \
    xz-devel \
    git \
    tmux
```
libcuda.s01

**Install pyenv** Instructions: https://github.com/pyenv/pyenv#basic-github-checkout

**Install python 3.7**
To install using pyenv
```bash
pyenv install 3.7.7
```
Then, from the repo root directory, set the python version:
```
pyenv local 3.7.7
```

**Setup virtual environment**
Install poetry
```bash
pip install poetry
```
Create venv and install python dependencies:
```bash
poetry install
```
Activate venv:
```bash
source .venv/bin/activate
```

