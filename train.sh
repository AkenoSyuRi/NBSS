#!/bin/bash

python SharedTrainer.py fit \
    --config=configs/onlineSpatialNet.yaml \
    --config=configs/datasets/librispeech.yaml \
    --model.channels="[0,1,2,3,4,5]" \
    --model.arch.dim_input=12 \
    --model.arch.dim_output=4 \
    --model.arch.num_freqs=257 \
    --trainer.precision=32 \
    --model.compile=true \
    --data.batch_size="[4,4]" \
    --trainer.devices=0, \
    --trainer.max_epochs=100
