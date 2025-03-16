#!/bin/bash

EXPNAME="TractOracleNet"
EXPID="testing_transformer"
MAXEP=100
# DATASET_FILE="/home/local/USHERBROOKE/levj1404/Documents/TractOracleNet/TractOracleNet/datasets/ismrm2015_1mm/train_test_classical_tracts_dataset.hdf5"
# DATASET_FILE="/home/local/USHERBROOKE/levj1404/Documents/TractOracleNet/TractOracleNet/datasets/ismrm2015_1mm/full_pft_ant.hdf5"
DATASET_FILE="/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/antoine-pft-eq-small-positive.hdf5"

python TractOracleNet/trainers/transformer_train.py \
    "experiments/${EXPNAME}" \
    ${EXPNAME} \
    ${EXPID} \
    ${MAXEP} \
    ${DATASET_FILE} \
    --lr 0.001 \
    --n_head 4 \
    --n_layers 4 \
    --batch_size 1024 \
    --num_workers 20
