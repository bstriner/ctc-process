import numpy as np
import pytest

from asr_vae.preprocessing import shard_files


def test_shards():
    n = 10
    shard_size = 3
    shards = shard_files(n, shard_size=shard_size)
    assert len(shards) == 4
    assert max(len(s) for s in shards) <= shard_size
    combo = np.concatenate(shards)
    sorted = np.sort(combo)
    assert np.all(sorted == np.arange(n))


if __name__ == '__main__':
    pytest.main([__file__])
