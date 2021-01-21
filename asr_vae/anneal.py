import tensorflow as tf


def calc_scale(min_val, max_val, start_step, end_step, step):
    step = tf.cast(step, tf.float32)
    start_step = tf.cast(start_step, tf.float32)
    end_step = tf.cast(end_step, tf.float32)
    val_mult = max_val - min_val
    step_mult = end_step - start_step
    offset = step - start_step
    scale = min_val + (offset * val_mult / step_mult)
    scale = tf.clip_by_value(scale, min_val, max_val)
    return scale


def get_scale_log(params):
    min_val = tf.log(params.anneal_min)
    max_val = tf.log(params.anneal_max)
    logscale = calc_scale(
        min_val=min_val,
        max_val=max_val,
        start_step=params.anneal_start,
        end_step=params.anneal_end,
        step=tf.train.get_or_create_global_step()
    )
    scale = tf.exp(logscale)
    tf.summary.scalar("anneal_scale", scale)
    return scale


def get_scale_linear(params):
    scale = calc_scale(
        min_val=params.anneal_min,
        max_val=params.anneal_max,
        start_step=params.anneal_start,
        end_step=params.anneal_end,
        step=tf.train.get_or_create_global_step()
    )
    tf.summary.scalar("anneal_scale", scale)
    return scale


def get_scale(params):
    if params.anneal_max == params.anneal_min:
        return params.anneal_max
    elif params.anneal_scale == 'log':
        return get_scale_log(params)
    elif params.anneal_scale == 'linear':
        return get_scale_linear(params)
    else:
        raise NotImplementedError()
