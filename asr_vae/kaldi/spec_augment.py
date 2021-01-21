import tensorflow as tf

from .sparse_image_warp import sparse_image_warp


# from tensorflow.contrib.image import sparse_image_warp

class SpecAugmentParams(object):
    def __init__(self, enabled, W, F, T, mF, mT, p):
        self.enabled = enabled
        self.W = W
        self.F = F
        self.T = T
        self.mF = mF
        self.mT = mT
        self.p = p

    @classmethod
    def from_params(cls, params):
        return cls(
            enabled=params.specaugment,
            W=params.specaugment_W,
            F=params.specaugment_F,
            T=params.specaugment_T,
            mF=params.specaugment_mF,
            mT=params.specaugment_mT,
            p=params.specaugment_p
        )


def augment(utterance, sa_params: SpecAugmentParams):
    if sa_params is None or not sa_params.enabled:
        return utterance
    W = tf.constant(sa_params.W, name="specaugment_W", dtype=tf.int32)
    F = tf.constant(sa_params.F, name="specaugment_F", dtype=tf.int32)
    p = tf.constant(sa_params.p, name="specaugment_p", dtype=tf.float32)
    T = tf.constant(sa_params.T, name="specaugment_T", dtype=tf.int32)

    h = utterance
    h = augment_warp(
        utterance=h,
        W=W
    )
    for _ in range(sa_params.mF):
        h = augment_freq_mask(
            utterance=h,
            F=F
        )
    for _ in range(sa_params.mT):
        h = augment_time_mask(
            utterance=h,
            T=T,
            p=p
        )
    with tf.control_dependencies(
            [tf.assert_equal(tf.shape(utterance), tf.shape(h))]
    ):
        h = tf.identity(h)
    h.set_shape(utterance.get_shape())
    return h


def augment_warp(utterance, W):
    """

    :param utterance: (ul, dim)
    :param utterance_length:
    :param W:
    :return:
    """
    X = tf.shape(utterance, out_type=tf.int32)[0]
    Y = tf.shape(utterance, out_type=tf.int32)[1]
    Y2 = tf.floordiv(Y, 2)
    Wmax = tf.minimum(W, tf.floordiv(X, 2) - 1)

    image = tf.expand_dims(tf.expand_dims(utterance, 0), -1)
    # with tf.control_dependencies([
    #    tf.assert_greater_equal(X, W*2, message='warp too big')
    # ]):
    srcX = tf.random.uniform(
        shape=[],
        minval=Wmax,
        maxval=X - Wmax,
        dtype=tf.int32
    )
    w = tf.random.uniform(
        shape=[],
        minval=1 - Wmax,
        maxval=Wmax,
        dtype=tf.int32
    )
    dstX = srcX + w
    zero = tf.constant(0, dtype=tf.int32)

    anchors = tf.stack(
        [
            tf.stack([zero, zero], 0),
            tf.stack([zero, Y], 0),
            tf.stack([X, Y], 0),
            tf.stack([X, zero], 0),
            tf.stack([zero, Y2], 0),
            tf.stack([X, Y2], 0)
        ], 0
    )
    src_pt = tf.expand_dims(tf.stack([srcX, Y2], 0), 0)
    dst_pt = tf.expand_dims(tf.stack([dstX, Y2], 0), 0)
    src_pts = tf.concat([src_pt, anchors], 0)
    dst_pts = tf.concat([dst_pt, anchors], 0)
    src_pts = tf.expand_dims(src_pts, 0)
    dst_pts = tf.expand_dims(dst_pts, 0)
    src_pts = tf.cast(src_pts, tf.float32)
    dst_pts = tf.cast(dst_pts, tf.float32)

    print("Image: {}".format(image))
    print("src_pts: {}".format(src_pts))
    print("dst_pts: {}".format(dst_pts))
    warped_image, _ = sparse_image_warp(
        image=image,
        source_control_point_locations=src_pts,
        dest_control_point_locations=dst_pts,
        interpolation_order=2,
        regularization_weight=0.0,
        num_boundary_points=0,
        name='sparse_image_warp'
    )
    print("Warped: {}".format(warped_image))
    warped_utterance = tf.squeeze(tf.squeeze(warped_image, -1), 0)
    return warped_utterance


def augment_freq_mask(utterance, F):
    Y = tf.shape(utterance)[1]
    f_delta = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=F + 1,
        dtype=tf.int32
    )
    f = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=Y - f_delta,
        dtype=tf.int32
    )
    mask = tf.concat([
        tf.ones(shape=(f,), dtype=utterance.dtype),
        tf.zeros(shape=(f_delta,), dtype=utterance.dtype),
        tf.ones(shape=(Y - f - f_delta,), dtype=utterance.dtype)
    ], axis=0)
    utterance_masked = utterance * mask
    return utterance_masked


def augment_time_mask(utterance, T, p):
    X = tf.shape(utterance)[0]
    Tmax = tf.minimum(T, tf.cast(tf.cast(X, p.dtype) * p, T.dtype))
    t_delta = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=Tmax + 1,
        dtype=tf.int32
    )
    t = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=X - t_delta,
        dtype=tf.int32
    )
    mask = tf.concat([
        tf.ones(shape=(t,), dtype=utterance.dtype),
        tf.zeros(shape=(t_delta,), dtype=utterance.dtype),
        tf.ones(shape=(X - t - t_delta), dtype=utterance.dtype)
    ], axis=0)
    mask = tf.expand_dims(mask, -1)
    utterance_masked = utterance * mask
    return utterance_masked
