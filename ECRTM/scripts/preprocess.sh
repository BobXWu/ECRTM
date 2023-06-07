#!/bin/bash
set -e

mkdir -p ../data/raw_data

python preprocess/download_20ng.py
python preprocess/preprocess.py -d ../data/raw_data/20ng/20ng_all --output_dir ../data/20NG --vocab-size 5000 --label group


python preprocess/download_imdb.py
python preprocess/preprocess.py -d ../data/raw_data/imdb --output_dir ../data/IMDB --vocab-size 5000 --label sentiment
