#!/usr/bin/env bash
singularity exec \
    /data/VOL3/bstriner/singularity/images/tf-nightly-cpu.simg \
    python3 -m tensorboard.main \
    --logdir /data/VOL3/bstriner/asr-vae/output/wsj \
    "$@" > /data/VOL3/bstriner/asr-vae/logs/tb.out 2>&1
