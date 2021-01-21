import os
import re
from collections import Counter

from asr_vae.kaldi.kaldi_records import read_transcripts

dirs = [
    'train_si284',
    'test_dev93',
    'test_eval92',
    'test_eval93'
]
for d in dirs:
    #transcripts_path = os.path.join('../../data/wsj/textdata', d, 'text-normalized')
    transcripts_path = os.path.join('../../data/wsj/textdata', d, 'text')
    transcripts = read_transcripts(transcripts_path)
    hyphenated = Counter()
    truncated = Counter()
    banged = Counter()
    coloned = Counter()
    nonword = Counter()
    wrongword = Counter()
    total = 0
    chars = set()
    for k, text in transcripts:
        w = text.split(" ")
        for x in w:
            if len(x)>0:
                if "-" in x:
                    i = x.index("-")
                    if i == 0 or i == len(x) - 1:
                        truncated.update([x])
                    else:
                        hyphenated.update([x])
                elif x[0]=="*" and x[-1]=="*":
                    wrongword.update([x])
                elif "!" in x:
                    banged.update([x])
                elif ":" in x:
                    coloned.update([x])
                elif not re.match("^[A-Z'\.]+$", x):
                    nonword.update([x])
                total += 1
                chars.update(x)
    chars=list(chars)
    chars.sort()
    print("Dir {}: {} total".format(d, total))
    print("Chars {}: {}".format(len(chars), chars))
    print("hyphenated {}: {}".format(sum(v for _, v in hyphenated.items()), hyphenated))
    print("wrongword {}: {}".format(sum(v for _, v in wrongword.items()), wrongword))
    print("nonword {}: {}".format(sum(v for _, v in nonword.items()), nonword))
    print("truncated {}: {}".format(sum(v for _, v in truncated.items()), truncated))
    print("banged {}: {}".format(sum(v for _, v in banged.items()), banged))
    print("coloned {}: {}".format(sum(v for _, v in coloned.items()), coloned))
    print()
    print()

