from asr_vae.kaldi.extract_vocab import extract_vocab

if __name__ == '__main__':
    file = '../../data/wsj/train_si284/text'
    extract_vocab(file, '../../data/wsj/vocab.npy')
