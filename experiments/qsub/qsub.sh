#!/usr/bin/env bash
export QSUB_NODES="$1"
shift
qsub \
    -m ea -M bstriner@cs.cmu.edu \
    -q gpu \
    -l ${QSUB_NODES} \
    /data/VOL3/bstriner/asr-vae/experiments/qsub/singularity.sh \
    "$@"
