# olympic-songbirds
 Avian biodiversity and vocalization behavior in Washington's Olympic Experimental State Forest



Follow setup instructions: https://github.com/kahst/BirdNET-Analyzer

```
source env/bin/activate

pip3 install --upgrade pip
pip3 install librosa resampy
pip3 install tensorflow
pip3 install birdnetlib

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda

conda config --add channels conda-forge

conda create -n birdnet-analyzer python=3.10 -c conda-forge -y
conda activate birdnet-analyzer
```

Run the classifier with classification/run_analyze_parallel.py