# avian-bioacoustics-oesf
 Avian biodiversity and vocalization behavior in Washington's Olympic Experimental State Forest

## Recommendations
- Visual Studio Code with Microsoft extensions [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy), and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- Install dependencies to a [virtual python environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) (.venv) that is used exclusively for this project. Create the environment manually, or via VS Code.

## Prerequisites
- [Python](https://www.python.org/downloads/) 3.9+ 64-bit (3.10 recommended, ensure "Add path to environment variables" is checked during install)

### Package dependencies
From a terminal shell within the virtual environment, navigate to the root directory of this repository (`.../avian-bioacoustics-oesf`), and run:

```
git submodule update --init --recursive
pip install --upgrade pip
pip install -r requirements.txt
```

> This should install all dependencies you need to run, for example, `analyze_file.py`. For further reference, see setup instructions for [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) and [birdnetlib](https://github.com/joeweiss/birdnetlib).

### Optional: Sound separation dependencies

This repository has a fork of [bird_mixit](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit) as a submodule, which is used to perform sound separation.
1. Follow steps to install [gsutil](https://cloud.google.com/storage/docs/gsutil_install)
1. Install bird_mixit model checkpoints via a terminal shell (see below)

```
gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints data/models/
```

## Contents
- `analyses` – ecological analyses and models (e.g. species accumulation curves)
- `annotation` – extracting samples for annotation and clustering
- `classification` – run classifier on audio/directory, evaluate performance
- `data` – data associated with the study area and project

## Training
```
cd src/submodules/BirdNET-Analyzer/
# Ensure you have training data located in the input directory and a valid output directory
python3 train.py --i /Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/training/2020_05_10 --o /Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/models/2020_05_10/custom_classifier.tflite
```
