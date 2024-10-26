# avian-bioacoustics-oesf
Few-shot transfer learning with BirdNET to enable monitoring of avian biodiversity and vocalization behavior in Washington's Olympic Experimental State Forest

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
- `data` – data associated with the study area and project
- `src` – all source code for data annotation, processing, and analysis
    - `annotation` – sample annotation interface
    - `classification` – internal wrapper code for `BirdNET-Analyzer` and `birdnetlib`
    - `R` – figures and ecological analysis in R
    - `submodules` – repository dependencies
    - `utils` – logging and helper modules


## Audio classification with BirdNET and/or custom model
Run `src/analyze_directory.py` or `src/analyze_file.py` to process model predictions for a given directory or file, respectively.

## Training and performance evaluation pipeline
Under development.
