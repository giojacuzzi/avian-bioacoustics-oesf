# avian-bioacoustics-oesf
 Avian biodiversity and vocalization behavior in Washington's Olympic Experimental State Forest

## Recommendations
- Visual Studio Code with Microsoft extensions [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy), and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- Install dependencies to a [virtual python environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) (.venv) that is used exclusively for this project. Create the environment manually, or via VS Code.

## Prerequisites
- [Python](https://www.python.org/downloads/) 3.9+ 64-bit (3.10 recommended, ensure "Add path to environment variables" is checked during install)

### Package dependencies
From a terminal shell within the virtual environment, run:

```
git submodule update --init --recursive
pip install --upgrade pip
pip install -r requirements.txt
```

> This should install all dependencies you need to run, for example, `analyze_file.py`. For further reference, see setup instructions for [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) and [birdnetlib](https://github.com/joeweiss/birdnetlib).

To install sound separation dependencies:
- Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install)
- Clone [bird_mixit](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit) repository

## Contents
- `analyses` – ecological analyses and models (e.g. species accumulation curves)
- `annotation` – extracting samples for annotation and clustering
- `classification` – run classifier on audio/directory, evaluate performance
- `data` – data associated with the study area and project
