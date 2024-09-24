from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kaleido
from kaleido import Tensor
from kaleido import FractalTensor

import random
random.seed(12345)

MIN_LEN = 4
MAX_LEN = 27

__all__ = [
    'gen_dataset',
    'gen_dataset_time_major',
    'gen_equal_length_seqs',
]


def gen_dataset(batch_size: int, vocab_size: int, device: str = 'cpu'):
    """Generate `batch_size` sequences that may have different lengths.

    - A batch of sequences is a 1-depth FractalTensor, analogous to
      List[List[int]].
    - A sequence is a 0-depth FractalTensor with integers as elements,
      analogous to List[int].
    """

    seqs = []
    for j in range(batch_size):
        seq_len = random.randint(MIN_LEN, MAX_LEN)
        seqs.append(
            [random.randint(0, vocab_size - 1) for i in range(seq_len)])
    return FractalTensor.from_pylist(seqs, device)


def gen_dataset_time_major(batch_size: int,
                           vocab_size: int,
                           device: str = 'cpu'):
    min_len = random.randint(MIN_LEN, MAX_LEN // 2)
    max_len = random.randint(min_len + 1, MAX_LEN)

    seq_lens = [random.randint(min_len, max_len) for i in range(batch_size)]
    seq_lens.sort(reverse=True)

    seqs = []
    for i in range(seq_lens[-1]):
        seqs.append(
            [random.randint(0, vocab_size - 1) for i in range(batch_size)])

    new_batch_size = batch_size - 1
    for j in range(batch_size - 2, -1, -1):
        num = seq_lens[j] - seq_lens[j + 1]
        for j in range(num):
            seqs.append([
                random.randint(0, vocab_size - 1)
                for i in range(new_batch_size)
            ])
        new_batch_size -= 1
    ft = FractalTensor.from_pylist(seqs, device)
    return ft


def gen_equal_length_seqs(batch_size: int,
                          vocab_size: int,
                          seq_len: int,
                          device: str = 'cpu'):
    """Generate `batch_size` sequences that may have different lengths.

    - A batch of sequences is a 1-depth FractalTensor, analogous to
      List[List[int]].
    - A sequence is a 0-depth FractalTensor with integers as elements,
      analogous to List[int].
    """

    seqs = []
    for j in range(batch_size):
        seqs.append(
            [random.randint(0, vocab_size - 1) for i in range(seq_len)])
    return FractalTensor.from_pylist(seqs, device)
