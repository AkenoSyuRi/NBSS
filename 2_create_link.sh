#!/bin/bash

# merge the librispeech and aishell3 dataset into one dst directory, process by creating soft link

libri_train="/home/featurize/data/audio_test/orig_dataset/LibriSpeech/train-clean-100"
libri_valid="/home/featurize/data/audio_test/orig_dataset/LibriSpeech/dev-clean"
libri_test="/home/featurize/data/audio_test/orig_dataset/LibriSpeech/test-clean"

aishell_train="/home/featurize/data/audio_test/orig_dataset/aishell3/train/wav"
aishell_valid="/home/featurize/data/audio_test/orig_dataset/aishell3/test/wav"

dst_train="/home/featurize/data/audio_test/libri_aishell/train"
dst_valid="/home/featurize/data/audio_test/libri_aishell/valid"
dst_test="/home/featurize/data/audio_test/libri_aishell/test"

# =========================================================
echo "link start"

for d in $(ls $libri_train); do ln -sf $libri_train/$d $dst_train; done
for d in $(ls $aishell_train); do ln -sf $aishell_train/$d $dst_train; done

for d in $(ls $libri_valid); do ln -sf $libri_valid/$d $dst_valid; done
for d in $(ls $aishell_valid); do ln -sf $aishell_valid/$d $dst_valid; done

for d in $(ls $libri_test); do ln -sf $libri_test/$d $dst_test; done
# for d in `ls $aishell_test`; do ln -sf $aishell_test/$d $dst_test; done

echo "link end"
