import os

import numpy as np


def read_ark(path):
    stack = []
    id = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            if line[-1] == '[':
                id = line[0]
            elif line[-1] == ']':
                data = [float(d) for d in line[:-1]]
                stack.append(data)
                yield id, np.array(stack, dtype=np.float32)
                id = None
                stack = []
            else:
                data = [float(d) for d in line]
                stack.append(data)


def extract_features(feats_file, path):
    os.makedirs(path, exist_ok=True)
    data = read_ark(feats_file)
    for id, datum in data:
        fout = os.path.join(path, '{}.npy'.format(id))
        np.save(fout, datum)
