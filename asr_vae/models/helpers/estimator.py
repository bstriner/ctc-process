import os

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.ops.ctc_ops import ctc_loss_v2
from tensorflow.python.training import basic_session_run_hooks

from ..callbacks.asr_hook import AsrHook
from ..callbacks.gradient_accumulator import GradientAccumulatorHook
from ..make_train_op import get_total_loss, make_train_op
from ..metrics.asr_metrics import EVAL_SUMMARIES, SLOW_SUMMARIES, asr_metrics_and_summaries, sparse_to_clean_strings
from ...models.make_train_op import make_opt, make_transform_grads_fn
from ...sparse import sparsify
from ...train_loop import make_learning_rate


def move_blank(inputs):
    outputs = tf.concat(
        [inputs[:, :, 1:], inputs[:, :, :1]],
        axis=-1
    )
    return outputs


def add_summary_ops(run_config, training_hooks, evaluation_hooks):
    eval_summary_op = tf.summary.merge(tf.get_collection(EVAL_SUMMARIES), collections=[])
    eval_summary_hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=1,
        output_dir=os.path.join(run_config.model_dir, 'eval'),
        summary_op=eval_summary_op)
    slow_summary_op = tf.summary.merge(tf.get_collection(SLOW_SUMMARIES), collections=[])
    slow_summary_hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=tf.app.flags.FLAGS.save_summary_steps_slow,
        output_dir=run_config.model_dir,
        summary_op=slow_summary_op)
    training_hooks.append(slow_summary_hook)
    evaluation_hooks.append(eval_summary_hook)


def calc_ctc_loss_n(transcripts, logits_t, transcript_lengths, transcript_idx, logit_lengths, params):
    if tf.app.flags.FLAGS.ctc_mode == 'dense':
        return ctc_loss_v2(
            labels=transcripts + 1,
            logits=logits_t,
            label_length=transcript_lengths,
            logit_length=logit_lengths,
            logits_time_major=True,
            unique=None,
            blank_index=0,
            name="ctc_loss_dense"
        )
    elif tf.app.flags.FLAGS.ctc_mode == 'sparse':
        transcript_values = tf.gather_nd(
            params=transcripts + 1,
            indices=transcript_idx
        )
        transcripts_sparse = tf.SparseTensor(
            values=transcript_values,
            indices=transcript_idx,
            dense_shape=tf.shape(transcripts, out_type=tf.int64)
        )
        return ctc_loss_v2(
            labels=transcripts_sparse,
            logits=logits_t,
            label_length=None,
            logit_length=logit_lengths,
            logits_time_major=True,
            unique=None,
            blank_index=0,
            name="ctc_loss_sparse"
        )
    else:
        raise ValueError()


def make_estimator(
        params,
        logits_t,
        glogits_t,
        logit_lengths,
        transcripts,
        transcript_lengths,
        mode,
        vocab,
        run_config: RunConfig,
        model_scope,
        encoder_scope,
        decoder_scope,
        utterance_id,
        is_training,
        training_hooks=None,
        eval_metric_ops=None,
        sentencepiece=False
):
    if training_hooks is None:
        training_hooks = []
    if eval_metric_ops is None:
        eval_metric_ops = {}
    tl = tf.shape(transcripts)[1]
    transcript_mask = tf.sequence_mask(lengths=transcript_lengths, maxlen=tl)
    transcript_idx = tf.where(transcript_mask)
    transcripts_sparse = sparsify(transcripts, transcript_mask)
    vocab_tensor = tf.constant(vocab, dtype=tf.dtypes.string)
    transcripts_strings = sparse_to_clean_strings(
        sparse=transcripts_sparse,
        vocab_table=vocab_tensor
    )

    assert glogits_t is not None
    tf.summary.histogram('ctc_logits', glogits_t, collections=[EVAL_SUMMARIES, SLOW_SUMMARIES])
    with tf.name_scope(model_scope + "/"):
        ctc_loss_n = calc_ctc_loss_n(
            transcripts=transcripts,
            logits_t=logits_t,
            transcript_lengths=transcript_lengths,
            logit_lengths=logit_lengths,
            params=params,
            transcript_idx=transcript_idx
        )
        ctc_loss = tf.reduce_mean(ctc_loss_n)
        ctc_loss = tf.check_numerics(ctc_loss, "ctc_loss check_numerics", name='ctc_loss_check_numerics')
        tf.summary.scalar('ctc_loss', ctc_loss)
        eval_metric_ops[model_scope + "/ctc_loss"] = tf.metrics.mean(ctc_loss_n)
        tf.losses.add_loss(ctc_loss)

    evaluation_hooks = []
    predictions = {
        "utterance_id": utterance_id,
        "transcripts": tf.sparse_tensor_to_dense(transcripts_sparse, default_value=-1),
        "transcript_strings": transcripts_strings
    }

    # Generated metrics
    if params.model in ['aae', 'vae', 'aae-stoch', 'ctc']:
        glogits_t_blank = tf.concat(
            [glogits_t[:, :, 1:], glogits_t[:, :, :1]],
            axis=-1
        )
        if 'random_search' in tf.flags.FLAGS and tf.flags.FLAGS.random_search:
            uniform = tf.random.uniform(
                shape=tf.shape(glogits_t_blank),
                minval=np.finfo(np.float_).tiny,
                maxval=1.,
                dtype=tf.float32
            )
            gumbel = -tf.math.log(-tf.math.log(uniform))
            glogits_t_blank += gumbel
        (gdecoded_sparse,), neg_sum_logits = tf.nn.ctc_beam_search_decoder(
            inputs=glogits_t_blank,
            sequence_length=logit_lengths,
            beam_width=tf.flags.FLAGS.beam_width,
            top_paths=1,
            merge_repeated=True
        )
        gdecoded_strings = sparse_to_clean_strings(
            sparse=gdecoded_sparse,
            vocab_table=vocab_tensor
        )
        predictions.update({
            "generated": tf.sparse_tensor_to_dense(gdecoded_sparse, default_value=-1),
            "generated_strings": gdecoded_strings
        })

        asr_metrics_and_summaries(
            transcripts_strings=transcripts_strings,
            transcripts_sparse=transcripts_sparse,
            decoded_strings=gdecoded_strings,
            decoded_sparse=gdecoded_sparse,
            eval_metric_ops=eval_metric_ops,
            prefix="",
            predictions=predictions,
            sentencepiece=sentencepiece
        )
        gasr_hook = AsrHook(
            true_strings=transcripts_strings,
            generated_strings=gdecoded_strings,
            path=os.path.join(run_config.model_dir, "generated", "generated-{:08d}.csv")
        )
        evaluation_hooks.append(gasr_hook)
    elif params.model in ['ae']:
        pass
    else:
        raise NotImplementedError()

    # Autoencoded metrics
    if params.model in ['aae', 'vae', 'aae-stoch', 'ae']:
        logits_t_blank = tf.concat(
            [logits_t[:, :, 1:], logits_t[:, :, :1]],
            axis=-1
        )
        # (decoded_sparse,), neg_sum_logits = tf.nn.ctc_greedy_decoder(
        #    inputs=logits_t_blank,
        #    sequence_length=logit_lengths
        # )
        (decoded_sparse,), neg_sum_logits = tf.nn.ctc_beam_search_decoder(
            inputs=logits_t_blank,
            sequence_length=logit_lengths,
            beam_width=tf.flags.FLAGS.beam_width,
            top_paths=1,
            merge_repeated=True)
        tf.summary.histogram('autoencoded_ctc_logits', logits_t, collections=[EVAL_SUMMARIES, SLOW_SUMMARIES])
        decoded_strings = sparse_to_clean_strings(
            sparse=decoded_sparse,
            vocab_table=vocab_tensor
        )
        predictions.update({
            "autoencoded": tf.sparse_tensor_to_dense(decoded_sparse, default_value=-1),
            "autoencoded_strings": decoded_strings
        })
        asr_metrics_and_summaries(
            transcripts_strings=transcripts_strings,
            transcripts_sparse=transcripts_sparse,
            decoded_strings=decoded_strings,
            decoded_sparse=decoded_sparse,
            eval_metric_ops=eval_metric_ops,
            prefix="autoencoded_",
            predictions=predictions,
            sentencepiece=sentencepiece
        )
        asr_hook = AsrHook(
            true_strings=transcripts_strings,
            generated_strings=decoded_strings,
            path=os.path.join(run_config.model_dir, "autoencoded", "autoencoded-{:08d}.csv")
        )
        evaluation_hooks.append(asr_hook)
    elif params.model == 'ctc':
        pass
    else:
        raise NotImplementedError()

    add_summary_ops(
        run_config=run_config,
        training_hooks=training_hooks,
        evaluation_hooks=evaluation_hooks)

    total_loss = get_total_loss(scope=model_scope)
    lr = make_learning_rate(
        lr=params.lr,
        eval_metric_ops=eval_metric_ops
    )
    if is_training:
        if params.train_acc == 1:
            train_op = make_train_op(
                scope=model_scope,
                lr=lr,
                params=params,
                global_step=tf.train.get_or_create_global_step(),
                total_loss=total_loss,
                opt=params.optimizer
            )
        elif params.train_acc > 1:
            acc_hook = GradientAccumulatorHook(
                loss=total_loss,
                var_list=tf.trainable_variables(scope=model_scope),
                opt=make_opt(lr=lr, params=params, opt=params.optimizer),
                global_step=tf.train.get_or_create_global_step(),
                frequency=params.train_acc,
                transform_grads_fn=make_transform_grads_fn(params=params)
            )
            train_op = acc_hook.train_op
            training_hooks.append(acc_hook)
        else:
            raise ValueError("train_acc: {}".format(params.train_acc))
        # if encoder_scope is not None:
        #    encoder_train_op = make_train_op(
        #        scope=encoder_scope.name,
        #        lr=params.encoder_lr,
        #        params=params,
        #        global_step=None,
        #        total_loss=total_loss
        #    )
        #    train_op = tf.group(train_op, encoder_train_op)
    else:
        train_op = None
    print("Returning estimatorspec")
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks,
        train_op=train_op,
        predictions=predictions,
        training_hooks=training_hooks)
