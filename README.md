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
Controlled via `/src/config.py`

## Run pipeline
From within the virtual environment:
```
(.venv) whatsappnalysis$ python src/pipeline.py
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
