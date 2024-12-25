#!/bin/bash

set -x

torchrun \
    --nnodes=1 --nproc-per-node 8 --node-rank 0 \
    --master-addr=127.0.0.1 --master-port=28888 \
    train.py \
    $@ 2>&1 | tee log.txt
