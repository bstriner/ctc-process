#!/usr/bin/env bash
QSUB_NODES="$1"
ARGUMENTS="$2"
qsub \
    -m ea -M bstriner@cs.cmu.edu \
    -q gpu \
    -l "${QSUB_NODES}" \
    /data/VOL3/bstriner/asr-vae/experiments/wsj/wsj_singularity.sh \
    -F "${ARGUMENTS}"
