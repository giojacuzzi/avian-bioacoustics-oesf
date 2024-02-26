Follow setup instructions: https://github.com/kahst/BirdNET-Analyzer

First time, install birdnetlib:
```
source env/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt

# macos
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda

conda config --add channels conda-forge

conda create -n birdnet-analyzer python=3.10 -c conda-forge -y
conda activate birdnet-analyzer
```

Get species list:
```
python3 species.py --o '~/repos/olympic-songbirds/data/species/species_eBird.txt' --lat 47.63610 --lon -124.37216 --week -1
```

Run analyzer to create selection table:
```
python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4

# Include --lat, --long, --week, and --overlap! Look into --sf_thresh
```

Install source-separation:
- Install gsutil https://cloud.google.com/storage/docs/gsutil_install
- Clone repo https://github.com/google-research/sound-separation/tree/master/models/bird_mixit


