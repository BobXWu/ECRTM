#!/bin/bash
set -e

model=${1}
dataset=${2}
K=${3:-50}
index=${4:-1}

echo ------ ${dataset} ${model} K=${K} ${index}th `date` ------

python run.py --model ${model} --dataset ${dataset} --config configs/model/${model}_${dataset}.yaml --num_topic ${K}

./scripts/eva.sh ${model} ${dataset} ${K} ${index}
