#!/usr/bin/env bash
source /data/VOL3/bstriner/pyvenv/bin/activate
python3.6 -m tensorboard.main \
    --logdir=/data/VOL3/bstriner/asr-vae/output/librispeech \
    > /data/VOL3/bstriner/asr-vae/logs/librispeech-tb.out \
    2>&1
