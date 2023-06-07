#!/bin/bash
set -e

model=${1}
dataset=${2}
K=${3:-50}
index=${4:-1}
T=15

prefix=./output/${dataset}/${model}_K${K}_${index}th
topic_path=${prefix}_T${T}

type=C_V

jar_dir=./palmetto

echo -n "===>${type}_T${T}: "

java -jar ${jar_dir}/palmetto-0.1.0-jar-with-dependencies.jar ${jar_dir}/wikipedia/wikipedia_bd ${type} ${topic_path} | tail -${K} | awk '{print($2)}' | awk '{sum+=$1} END {print sum/NR}'

python utils/eva/TD.py --data_path ${topic_path}

python utils/eva/cluster.py --path ${prefix}_params.mat --label_path ../data/${dataset}
