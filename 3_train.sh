#!/bin/bash

# generate RIRs
# python generate_rirs.py --mic_zlim="[2.5, 3.5]" --rir_nums="[50000, 5000, 3000]" --rir_dir=~/data/audio_test/orig_dataset/nbss_rirs_generated

# train
# python SharedTrainer.py fit \
#     --config=configs/onlineSpatialNet.yaml \
#     --config=configs/datasets/librispeech.yaml \
#     --model.channels="[0,1,2,3,4,5]" \
#     --model.arch.dim_input=12 \
#     --model.arch.dim_output=4 \
#     --model.arch.num_freqs=257 \
#     --trainer.precision=32 \
#     --model.compile=true \
#     --data.batch_size="[4,4]" \
#     --trainer.devices=0, \
#     --trainer.max_epochs=100

# train data-driven
python SharedTrainer.py fit \
    --config=configs/onlineSpatialNet.yaml \
    --config=configs/datasets/librispeech.yaml \
    --model.channels="[0,1,2,3,4,5,6]" \
    --model.arch.dim_input=14 \
    --model.arch.dim_output=2 \
    --model.arch.num_freqs=257 \
    --model.compile=true \
    --model.loss.loss_func="models.io.loss.cc_mse" \
    --model.loss.pit=false \
    --trainer.precision=32 \
    --data.batch_size="[4,4]" \
    --trainer.devices=0, \
    --trainer.max_epochs=100

# resume
# python SharedTrainer.py fit \
#     --ckpt_path=~/output/nbss_logs/OnlineSpatialNet/version_3/checkpoints/last.ckpt \
#     --config=configs/onlineSpatialNet.yaml \
#     --config=configs/datasets/librispeech.yaml \
#     --model.channels="[0,1,2,3,4,5,6]" \
#     --model.arch.dim_input=14 \
#     --model.arch.dim_output=2 \
#     --model.arch.num_freqs=257 \
#     --model.compile=true \
#     --model.loss.loss_func="models.io.loss.cc_mse" \
#     --model.loss.pit=false \
#     --trainer.precision=32 \
#     --data.batch_size="[4,4]" \
#     --trainer.devices=0, \
#     --trainer.max_epochs=100
