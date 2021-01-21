
DIM=512
DEPTH=6
LR=3e-4
BATCHNORM=batch_norm
VARIATIONAL=True
VARIATIONAL_MEAN=False
VARIATIONAL_LEARN_PRIOR=True
DROPOUT=0.1
UOUT=True
JOB_NAME="wsj-dim${DIM}-depth${DEPTH}-lr${LR}-bn${BATCHNORM}-v${VARIATIONAL}-vm${VARIATIONAL_MEAN}-vlp${VARIATIONAL_LEARN_PRIOR}-prior0.1-do${DROPOUT}${UOUT}"
echo "${JOB_NAME}"
sbatch \
    --job-name="$JOB_NAME" \
    --partition=2gpu \
    --nodelist=islpc38 \
    --mem=10G \
    --mail-type=ALL \
    --mail-user=bstriner@cs.cmu.edu \
    --gres=gpu:1 \
    --time=48:00:00 \
<<EOF
#!/bin/bash
source /data/VOL3/bstriner/pyvenv-islpc/bin/activate
source /data/VOL3/bstriner/cuda/cuda-10.0/activate
export PYTHONPATH=/data/VOL3/bstriner/asr-vae
cd /data/VOL3/bstriner/asr-vae/experiments/wsj
python3.6 /data/VOL3/bstriner/asr-vae/experiments/wsj/wsj_train.py \
    --config="conf/wsj_ctc_variational.json" \
    --model_dir="../../output/wsj/ctc-variational-v3/${JOB_NAME}" \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --save_summary_steps=100 \
    --save_summary_steps_slow=400 \
    --save_checkpoints_steps=2000 \
    --max_steps=500000 \
    --max_steps_without_decrease=50000 \
    --hparams="decoder_uout=${UOUT},decoder_dropout=${DROPOUT},variational=${VARIATIONAL},variational_sigma_prior=0.1,variational_mean=${VARIATIONAL_MEAN},variational_learn_prior=${VARIATIONAL_LEARN_PRIOR},clip_gradient_value=0.1,clip_gradient_norm=0.0,decoder_dim=${DIM},decoder_depth=${DEPTH},lr=${LR},batch_norm=${BATCHNORM},batch_norm_scale=True" \
    --train_data_dir="/scratch/bstriner/wsj-data/tfrecords/train_si284" \
    --eval_data_dir="/scratch/bstriner/wsj-data/tfrecords/test_dev93" \
    > "/data/VOL3/bstriner/asr-vae/logs/variational-v3/${JOB_NAME}.txt" \
    2>&1
EOF

