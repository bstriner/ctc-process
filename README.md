# ctc-process

Train an ASR system using a combination of CTC and VAE that
allows non-autoregressive end-to-end prediction.

## Install

Run `python setup.py develop`

## Dataprep

Use the DNN preconfiguration from Kaldi to preprocess data and extract audio features from the datasets.

For WSJ, you will need to download the following corpora:

- LDC93S6B
- LDC94S13B

## Training

Run `wsj_train.py` or `librispeech_train.py`

- `--config` configuration file see example files in `conf` directory.
- `--model-dir` directory for saving/loading/resuming model data
- `--train-data-dir` directory containing training data
- `--eval-data-dir` directory containing evaluation data




