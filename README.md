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

## Training and performance evaluation pipeline
1. Manually annotate training examples with Raven Pro
2. Run `training_extract_audio_examples.py` to extract audio examples for training
    - Manually add any additonal class examples (e.g. "Background") to the `audio` subdirectory 
3. Run `training_assemble_datasets.py` to assemble datasets for training
4. Train custom classifiers with TODO
```
cd src/submodules/BirdNET-Analyzer/
python3 train_custom.py --i /Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/models/custom/custom_S1_N2_A0/training_files.csv --o /Users/giojacuzzi/repos/avian-bioacoustics-oesf/data/models/custom/custom_S1_N2_A0/custom_S1_N2_A0.tflite --no-autotune
```
4. Return to top directory with `cd ../../../`, then run `test_compare_validation_performance.py`

