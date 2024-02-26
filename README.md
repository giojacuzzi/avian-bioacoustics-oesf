# avian-bioacoustics-oesf
 Avian biodiversity and vocalization behavior in Washington's Olympic Experimental State Forest

## Recommendations
- Visual Studio Code with Microsoft extensions [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy), and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- I recommend installing dependencies to a [virtual python environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) (.venv) that is used exclusively for this project.

## Prerequisites
- [Python](https://www.python.org/downloads/) 3.9+ 64-bit (3.10 recommended, ensure "Add path to environment variables" is checked during install)
- [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer)
- [birdnetlib](https://github.com/joeweiss/birdnetlib)

### Package dependencies
Follow setup instructions for [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) and [birdnetlib](https://github.com/joeweiss/birdnetlib). Steps are shown below:

```
pip3 install --upgrade pip
pip3 install pandas
pip3 install librosa resampy
pip3 install tensorflow # 2.5 or later, may need to enable "long paths" on Windows
pip3 install birdnetlib
```

To install sound separation dependencies:
- Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install)
- Clone [bird_mixit](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit) repository

## Contents
- `analyses` – ecological analyses and models (e.g. species accumulation curves)
- `annotation` – extracting samples for annotation and clustering
- `classification` – run classifier on audio/directory, evaluate performance
- `data` – data associated with the study area and project
