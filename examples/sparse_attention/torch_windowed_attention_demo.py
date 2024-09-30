import torch
import torch.nn.functional as F


def test(qss, kss, vss):
    # Q, K, V: [batch_size, block_num, block_size, hidden]
    Q = qss[:, :, 2:-2:, :]
    K = torch.cat((kss[:, :, 1:-3, :], kss[:, :, 2:-2, :], kss[:, :, 3:-1, :]),
                  2)
    V = torch.cat((vss[:, :, 1:-3, :], vss[:, :, 2:-2, :], vss[:, :, 3:-1, :]),
                  2)
    QK = torch.einsum("blqd,blkd->blqk", Q, K)
    attn_weights = F.softmax(QK, -1)
    attn_vecs = torch.einsum("blqk,blkd->blqd", attn_weights, V)
    return attn_vecs


if __name__ == '__main__':
    batch_size = 16
    block_size = 16
    seq_len = 512
    hidden_dim = 128
    block_num = seq_len // block_size

    queries = torch.rand(batch_size, block_num, block_size, hidden_dim)
    keys = torch.rand(batch_size, block_num, block_size, hidden_dim)
    values = torch.rand(batch_size, block_num, block_size, hidden_dim)

    test(queries, keys, values)
