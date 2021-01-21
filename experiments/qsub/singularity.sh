#!/usr/bin/env bash
export COMPUTE_NAME=`hostname | sed 's/\..*//'`
singularity exec --nv /data/VOL3/bstriner/singularity/images/${COMPUTE_NAME}.simg "$@"
