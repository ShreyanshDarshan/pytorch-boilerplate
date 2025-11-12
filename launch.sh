#!/bin/bash
NUM_GPUS=1
NUM_NODES=1
config="configs/polynomial.gin"

if [ $NUM_NODES -eq 1 ]; then
    torchrun \
        --standalone \
        --nnodes=${NUM_NODES} \
        --nproc-per-node=${NUM_GPUS} \
        train.py --ginc ${config}
else
    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc-per-node=${NUM_GPUS} \
        --node-rank=0 \
        --master-addr=192.168.0.165 \
        --master-port=1234 \
        train.py --ginc ${config}
fi
