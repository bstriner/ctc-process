#!/usr/bin/env bash
singularity exec --nv \
    /data/VOL3/bstriner/singularity/images/10.0-tf-nightly.simg \
    /data/VOL3/bstriner/asr-vae/experiments/wsj/wsj_train.sh \
    "$@"
