import tensorflow as tf

NUM_MEL_BINS = 64
SAMPLE_RATE = 16000


def calc_mel(audio):
    stfts = tf.contrib.signal.stft(
        tf.reshape(audio, (1, -1)),
        frame_length=400,
        frame_step=160,
        fft_length=1024)
    magnitude_spectrograms = tf.abs(stfts)

    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 300.0, 4000.0
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS, num_spectrogram_bins, SAMPLE_RATE, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
    log_mel_spectrograms = tf.squeeze(log_mel_spectrograms, 0)
    return log_mel_spectrograms
