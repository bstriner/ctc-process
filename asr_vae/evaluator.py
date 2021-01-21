import csv
import json
import os

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig

from .default_params import load_hparams
from .kaldi.inputs import make_input_fn
from .kaldi.transcripts import decode_transcript
from .trainer import make_model_fn


def combine_predictions(predictions, output_path, vocab):
    preds = {}
    for prediction in predictions:
        utterance_id = prediction['utterance_id']
        if utterance_id not in preds:
            preds[utterance_id] = []
        preds[utterance_id].append(prediction)
    selected = []
    for k, ps in preds.items():
        counts = {}
        for p in ps:
            generated = decode_transcript(p['generated'], vocab)
            if generated not in counts:
                counts[generated] = [0, p]
            counts[generated][0] += 1
        counts = list(counts.items())
        counts.sort(key=lambda x: x[1][0])
        mode = counts[-1]
        selected.append(mode[1][1])
    return write_predictions(
        predictions=selected,
        output_path=output_path,
        vocab=vocab
    )


def combine_predictions_greedy(predictions, output_path, vocab):
    preds = {}
    for prediction in predictions:
        utterance_id = prediction['utterance_id']
        if utterance_id not in preds:
            preds[utterance_id] = []
        preds[utterance_id].append(prediction)
    selected = []
    for k, ps in preds.items():
        ps.sort(key=lambda x: x['word_edit_distance'])
        selected.append(ps[0])
    return write_predictions(
        predictions=selected,
        output_path=output_path,
        vocab=vocab
    )


def write_predictions(predictions, output_path, vocab):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'Idx',
            'True',
            'Generated',
            'True Character Count',
            'Hypothesis Character Count',
            'Character Edit Distance',
            'Character Error Rate',
            'True Word Count',
            'Hypothesis Word Count',
            'Word Edit Distance',
            'Word Error Rate'
        ])
        idx = 0
        tot_char_count_true = 0
        tot_char_count_hyp = 0
        tot_char_dist = 0
        tot_word_count_true = 0
        tot_word_count_hyp = 0
        tot_word_dist = 0

        for prediction in predictions:
            # print("Prediction: {}".format(prediction))
            transcripts = decode_transcript(prediction["transcripts"], vocab)
            generated = decode_transcript(prediction["generated"], vocab)
            character_count_true = prediction["character_count_true"]
            character_count_hypothesis = prediction["character_count_hypothesis"]
            character_edit_distance = prediction["character_edit_distance"]
            character_error_rate = prediction["character_error_rate"]
            word_count_true = prediction["word_count_true"]
            word_count_hypothesis = prediction["word_count_hypothesis"]
            word_edit_distance = prediction["word_edit_distance"]
            word_error_rate = prediction["word_error_rate"]
            tot_char_count_true += character_count_true
            tot_char_count_hyp += character_count_hypothesis
            tot_char_dist += character_edit_distance
            tot_word_count_true += word_count_true
            tot_word_count_hyp += word_count_hypothesis
            tot_word_dist += word_edit_distance
            w.writerow([
                idx,
                transcripts,
                generated,
                character_count_true,
                character_count_hypothesis,
                character_edit_distance,
                character_error_rate,
                word_count_true,
                word_count_hypothesis,
                word_edit_distance,
                word_error_rate
            ])
            idx += 1
        w.writerow([
            'Total',
            '',
            '',
            tot_char_count_true,
            tot_char_count_hyp,
            tot_char_dist,
            tot_char_dist / tot_char_count_true,
            tot_word_count_true,
            tot_word_count_hyp,
            tot_word_dist,
            tot_word_dist / tot_word_count_true
        ])
        print("Char rate: {}".format(tot_char_dist / tot_char_count_true))
        print("Word rate: {}".format(tot_word_dist / tot_word_count_true))


def evaluate():
    model_dir = tf.flags.FLAGS.model_dir
    os.makedirs(model_dir, exist_ok=True)
    # print("model_dir={}".format(model_dir))
    run_config = RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=tf.flags.FLAGS.save_checkpoints_steps)
    hparams = load_hparams(model_dir)

    with open(tf.app.flags.FLAGS.data_config, 'r') as fp:
        data_config = json.load(fp)
        input_dim = data_config['input_dim']
        vocab = data_config['vocab']
        vocab = np.array(vocab, dtype=np.unicode)
        sentencepiece = data_config['sentencepiece']
    if hparams.sentencepiece_online:
        spmodel = os.path.join(
            os.path.dirname(os.path.abspath(tf.app.flags.FLAGS.data_config)),
            "sentencepiece-model.model")
        sp = spm.SentencePieceProcessor()
        sp.LoadOrDie(spmodel)
    else:
        sp = None

    # Test Data
    eval_input_fn = make_input_fn(
        tf.flags.FLAGS.eval_data_dir,
        batch_size=tf.flags.FLAGS.eval_batch_size,
        shuffle=False,
        num_epochs=tf.flags.FLAGS.random_search_iter,
        subsample=hparams.subsample,
        average=True,
        independent_subsample=hparams.independent_subsample,
        bucket=hparams.bucket,
        bucket_size=hparams.bucket_size,
        buckets=hparams.buckets,
        sa_params=None,
        input_dim=input_dim,
        sp=sp)

    # Model
    model_fn = make_model_fn(run_config=run_config, vocab=vocab, sentencepiece=sentencepiece)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    predictions = estimator.predict(
        input_fn=eval_input_fn
    )
    if tf.flags.FLAGS.random_search_iter > 1:
        combine_predictions(predictions=predictions, output_path=tf.flags.FLAGS.prediction_path, vocab=vocab)
    else:
        write_predictions(predictions=predictions, output_path=tf.flags.FLAGS.prediction_path, vocab=vocab)
