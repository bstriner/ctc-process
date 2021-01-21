#!/usr/bin/env bash
sbatch --time=48:00:00 \
    --job-name="librispeech-tensorboard" \
    --partition=cpu \
    --mail-type=ALL \
    --mail-user=bstriner@cs.cmu.edu \
    /data/VOL3/bstriner/asr-vae/experiments/librispeech/librispeech_tensorboard.sh
