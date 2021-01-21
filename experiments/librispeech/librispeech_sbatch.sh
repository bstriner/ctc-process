#!/usr/bin/env bash
JOB_NAME="$1"
shift
sbatch \
    --job-name="$JOB_NAME" \
    --partition=gpu \
    --mail-type=ALL \
    --mail-user=bstriner@cs.cmu.edu \
    --gres=gpu:1 \
    --time=48:00:00 \
    /data/VOL3/bstriner/asr-vae/experiments/librispeech/librispeech_singularity.sh \
    "$@"
