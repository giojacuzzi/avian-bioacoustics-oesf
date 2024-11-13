# few-shot-transfer-learning-bioacoustics
Few-shot transfer learning with BirdNET to enable acoustic monitoring of wildlife communities.

### About

This repository contains the open source software, workflow routines, and reproducible methods used in the research article:

> Jacuzzi G., Olden J.D. Few-shot transfer learning enables robust acoustic community monitoring at the landscape scale. ** (in press).

Although this software was developed for a particular study region and set of monitoring objectives (avian biodiversity surveys in Washington's Olympic Experimental State Forest), it is designed to be freely repurposed and we encourage its use in other applications. Please cite the original publication in your references and direct any correspondance to gioj@uw.edu.

### Quickstart: GUI application

To run the bundled app executable:
> TODO: Final distribution not yet available

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

This installs all dependencies you need to run, for example,`src/gui.py` or `src/process_audio.py`. For further reference, see setup instructions for [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) and [birdnetlib](https://github.com/joeweiss/birdnetlib) if needed.

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
    - `audio` – internal audio processing and wrapper code for `BirdNET-Analyzer` and `birdnetlib`
    - `R` – figures and ecological analysis in R
    - `submodules` – repository dependencies
    - `utils` – logging and helper modules

## Audio classification with BirdNET and/or custom model
Run `src/process_audio.py` to process model predictions for a given directory or file. Show arguments with `python src/process_audio.py -h`, or see the script directly.

Alternatively, run the graphical user interface application at `src/gui.py` (see details below).

### GUI application packaging
To build and package the GUI as an executable from source:
- From console, activate the virtual environment
- `cd` navigate to repository root directory
- `pyinstaller gui.spec --clean`

## Training and performance evaluation pipeline
> TODO: under development.
