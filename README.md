WhatsAppnalysis
----

Load and analyze WhatsApp chat history, e.g. as exported to a text file.

# Developer setup

## Prerequisites
- [`poetry`](https://python-poetry.org/)
- python 3.7 environment, e.g. using [`pyenv`](https://github.com/pyenv/pyenv)

## Setup

Create virtual env. From the project root, run: 
```shell script
poetry install
```

Activate the venv
```shell script
source .venv/bin/activate
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
python src/pipeline.py
```


