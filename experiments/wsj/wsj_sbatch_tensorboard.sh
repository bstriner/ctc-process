#!/usr/bin/env bash
sbatch --time=48:00:00 \
    --job-name="wsj-tensorboard" \
    --partition=cpu \
    --mem-per-cpu=16G \
    --mail-type=ALL \
    --mail-user=bstriner@cs.cmu.edu \
    /data/VOL3/bstriner/asr-vae/experiments/wsj/wsj_tensorboard.sh
