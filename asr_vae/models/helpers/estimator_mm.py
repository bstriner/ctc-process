import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig

from .estimator import add_summary_ops, calc_ctc_loss_n, move_blank
from ..callbacks.gradient_accumulator import GradientAccumulatorHook
from ..make_train_op import get_total_loss, make_train_op
from ..metrics.asr_metrics import EVAL_SUMMARIES, SLOW_SUMMARIES, asr_metrics_and_summaries, sparse_to_clean_strings
from ...models.make_train_op import make_opt, make_transform_grads_fn
from ...sparse import sparsify
from ...train_loop import make_learning_rate


def make_estimator_mm(
        params,
        logits_t,
        gate_logits,
        logit_lengths,
        # glogits_t,
        # ggate_logits,
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
        evaluation_hooks=None,
        eval_metric_ops=None,
        predictions=None,
        sentencepiece=False
):
    if training_hooks is None:
        training_hooks = []
    if eval_metric_ops is None:
        eval_metric_ops = {}
    if evaluation_hooks is None:
        evaluation_hooks = []
    if predictions is None:
        predictions = {}

    n = tf.shape(transcripts)[0]
    tl = tf.shape(transcripts)[1]

    transcript_mask = tf.sequence_mask(lengths=transcript_lengths, maxlen=tl)
    transcript_idx = tf.where(transcript_mask)
    transcripts_sparse = sparsify(transcripts, transcript_mask)
    vocab_tensor = tf.constant(vocab, dtype=tf.dtypes.string)
    transcripts_strings = sparse_to_clean_strings(
        sparse=transcripts_sparse,
        vocab_table=vocab_tensor
    )

    mm_size = params.mm_size
    split_logits = [logits_t[:, :, i, :] for i in range(mm_size)]

    with tf.name_scope(model_scope + "/ctc_losses/"):
        decoded_mm_strings = []
        ctc_losses = []
        for i, logits_mm in enumerate(split_logits):
            with tf.name_scope('ctc_loss_{}'.format(i)):
                ctc_loss_n = calc_ctc_loss_n(
                    transcripts=transcripts,
                    logits_t=logits_mm,
                    transcript_lengths=transcript_lengths,
                    logit_lengths=logit_lengths,
                    params=params,
                    transcript_idx=transcript_idx
                )
                assert ctc_loss_n.shape.ndims == 1
                ctc_losses.append(ctc_loss_n)
                logits_mm_blank = move_blank(logits_mm)
                if 'random_search' in tf.flags.FLAGS and tf.flags.FLAGS.random_search:
                    uniform = tf.random.uniform(
                        shape=tf.shape(logits_mm_blank),
                        minval=np.finfo(np.float_).tiny,
                        maxval=1.,
                        dtype=tf.float32
                    )
                    gumbel = -tf.math.log(-tf.math.log(uniform))
                    logits_mm_blank += gumbel
                (decoded_mm,), _ = tf.nn.ctc_beam_search_decoder(
                    inputs=logits_mm_blank,
                    sequence_length=logit_lengths,
                    beam_width=tf.flags.FLAGS.beam_width,
                    top_paths=1,
                    merge_repeated=True
                )
                decoded_mm_string = sparse_to_clean_strings(
                    sparse=decoded_mm,
                    vocab_table=vocab_tensor
                )
                assert decoded_mm_string.shape.ndims == 1
                decoded_mm_strings.append(decoded_mm_string)
        ctc_losses = tf.stack(ctc_losses, axis=-1)  # (n, mm)
        decoded_mm_strings = tf.stack(decoded_mm_strings, axis=-1)  # (n, mm)
        assert gate_logits.shape.ndims == 2
        assert gate_logits.shape[-1].value == mm_size
        gate_logits = tf.nn.log_softmax(gate_logits, axis=-1)  # (n, mm)
        gating_probs = tf.exp(gate_logits)
        gating_probs_strings = tf.strings.as_string(gating_probs)

        with tf.name_scope('loss_calc'):
            weighted_loss_n = gate_logits - ctc_losses
            weighted_loss_n = -tf.reduce_logsumexp(weighted_loss_n, axis=-1, name='weighted_loss')
            eval_metric_ops['weighted_loss'] = tf.metrics.mean(weighted_loss_n)
            weighted_loss = tf.reduce_mean(weighted_loss_n)
            tf.losses.add_loss(weighted_loss)
            tf.summary.scalar('weighted_loss', weighted_loss)

    string_table = tf.concat(
        [tf.expand_dims(transcripts_strings, 1), decoded_mm_strings, gating_probs_strings],
        axis=-1
    )
    tf.summary.text("transcripts_mm", string_table, collections=[SLOW_SUMMARIES, EVAL_SUMMARIES])

    if 'random_search' in tf.flags.FLAGS and tf.flags.FLAGS.random_search:
        uniform = tf.random.uniform(
            shape=tf.shape(gate_logits),
            minval=np.finfo(np.float_).tiny,
            maxval=1.,
            dtype=tf.float32
        )
        gumbel = -tf.math.log(-tf.math.log(uniform))
        gating_selected = tf.argmax(gate_logits + gumbel, axis=-1, output_type=tf.int32)  # (n,)
    else:
        gating_selected = tf.argmax(gate_logits, axis=-1, output_type=tf.int32)  # (n,)

    logits = tf.transpose(logits_t, [1, 2, 0, 3])  # (n, mm, tl, v)
    logits_selected = tf.gather_nd(
        params=logits,
        indices=tf.stack([
            tf.range(n, dtype=tf.int32),
            gating_selected
        ], axis=-1))  # (n, tl, v)
    logits_selected = tf.transpose(logits_selected, [1, 0, 2])  # (tl, n, v)
    logits_selected = move_blank(logits_selected)
    if 'random_search' in tf.flags.FLAGS and tf.flags.FLAGS.random_search:
        uniform = tf.random.uniform(
            shape=tf.shape(logits_selected),
            minval=np.finfo(np.float_).tiny,
            maxval=1.,
            dtype=tf.float32
        )
        gumbel = -tf.math.log(-tf.math.log(uniform))
        logits_selected += gumbel
    (decoded_selected,), neg_sum_logits_selected = tf.nn.ctc_beam_search_decoder(
        inputs=logits_selected,
        sequence_length=logit_lengths,
        beam_width=tf.flags.FLAGS.beam_width,
        top_paths=1,
        merge_repeated=True
    )
    decoded_selected_strings = sparse_to_clean_strings(
        sparse=decoded_selected,
        vocab_table=vocab_tensor
    )
    asr_metrics_and_summaries(
        transcripts_strings=transcripts_strings,
        transcripts_sparse=transcripts_sparse,
        decoded_strings=decoded_selected_strings,
        decoded_sparse=decoded_selected,
        eval_metric_ops=eval_metric_ops,
        prefix="",
        predictions=predictions,
        sentencepiece=sentencepiece
    )

    predictions["transcripts"] = tf.sparse_tensor_to_dense(transcripts_sparse, default_value=-1)
    predictions["transcript_strings"] = transcripts_strings
    predictions["generated"] = tf.sparse_tensor_to_dense(decoded_selected, default_value=-1)
    predictions["generated_strings"] = decoded_selected_strings
    predictions['utterance_id'] = utterance_id

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
