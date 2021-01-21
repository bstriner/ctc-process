import os

from asr_vae.preprocessing import preprocess


def main():
    input_dir = r'D:\Projects\data\LibriSpeech\extracted'
    output_dir = '../dataset'
    names = [
        'train-clean-360',
        'train-clean-100',
        'dev-clean',
        'test-clean'
    ]
    vocab = None
    for name in names:
        indir = os.path.join(input_dir, name)
        odir = os.path.join(output_dir, name)
        vocab = preprocess(
            input_dir=indir,
            output_dir=odir,
            vocab=vocab
        )


if __name__ == '__main__':
    main()
