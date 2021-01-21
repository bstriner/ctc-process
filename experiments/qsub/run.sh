#!/usr/bin/env bash
cd /data/VOL3/bstriner/asr-vae/experiments/wsj
export PYTHONPATH=/data/VOL3/bstriner/asr-vae:$PYTHONPATH
python3 wsj_train.py \
        --config='conf/wsj_ctc.json' \
        --model_dir='../../output/wsj/ctc/v1' \
        --train_batch_size=32 \
        --eval_batch_size=32 \
        --save_summary_steps=200  \
        --save_checkpoints_steps=2000 > wsj_train_ctc.out 2>&1
