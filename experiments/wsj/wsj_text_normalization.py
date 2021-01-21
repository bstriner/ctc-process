import os

from asr_vae.kaldi.kaldi_records import read_transcripts, write_transcripts
from asr_vae.kaldi.text_normalization import normalize_sentence_for_training

dirs = [
    'train_si284',
    'test_dev93',
    'test_eval92',
    'test_eval93'
]
for d in dirs:
    transcripts_path = os.path.join('../../data/wsj/textdata', d, 'text')
    transcripts_normalized_path = os.path.join('../../data/wsj/textdata', d, 'text-normalized')
    transcripts = read_transcripts(transcripts_path)
    normalized_transcripts = (
        (k, normalize_sentence_for_training(v)) for k, v in transcripts
    )
    write_transcripts(
        transcripts_path=transcripts_normalized_path,
        transcripts=normalized_transcripts
    )
