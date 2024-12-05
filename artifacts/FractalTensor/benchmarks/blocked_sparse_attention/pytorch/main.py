from attention import BigbirdBlockSpareAttention
import torch
import argparse
import types


def test_BigBird(batch_size: int,
                 size_per_head: int,
                 from_seq_length: int,
                 from_block_size: int,
                 to_seq_length: int,
                 to_block_size: int,
                 num_attention_heads: int = 1,
                 num_rand_blocks: int = 3,
                 device='cuda:0',
                 output_file=None):
    query_layer = torch.rand(
        batch_size,
        num_attention_heads,
        from_seq_length,
        size_per_head,
        device=device)
    key_layer = torch.rand(
        batch_size,
        num_attention_heads,
        to_seq_length,
        size_per_head,
        device=device)
    value_layer = torch.rand(
        batch_size,
        num_attention_heads,
        to_seq_length,
        size_per_head,
        device=device)

    # The values should be 1 or 0. The attention scores will effectively be
    # set to -infinity for any positions in the mask that are 0, and will be
    # unchanged for positions that are 1.
    band_mask = torch.rand(
        batch_size,
        1,
        from_seq_length // from_block_size - 4,
        from_block_size,
        3 * to_block_size,
        device=device)
    from_mask = torch.rand(batch_size, 1, from_seq_length, 1, device=device)
    to_mask = torch.rand(batch_size, 1, 1, to_seq_length, device=device)
    from_blocked_mask = torch.rand(
        batch_size,
        from_seq_length // from_block_size,  # number blocks
        from_block_size,
        device=device)
    to_blocked_mask = torch.rand(
        batch_size,
        to_seq_length // to_block_size,  # number blocks
        to_block_size,
        device=device)

    attn = BigbirdBlockSpareAttention(
        num_attention_heads=num_attention_heads,
        num_rand_blocks=num_rand_blocks,
        size_per_head=size_per_head,
        from_block_size=from_block_size,
        to_block_size=to_block_size).to(device)

    attn(
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        batch_size,
        from_seq_length,
        to_seq_length,
        output_file,
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v in ('True'):
        return True
    elif v in ('False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_test_args():
    parser = argparse.ArgumentParser(description='Bigbird')
    parser.add_argument(
        '--seq_len', type=int, help='Sequence length', default=4096)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument(
        '--hidden_size', type=int, help='Hidden size', default=512)
    parser.add_argument(
        '--block_size', type=int, help='Block size', default=64)
    parser.add_argument(
        '--output_file', type=str, help='Output file path', default=None)
    parser.add_argument(
        '--default_test',
        type=str2bool,
        help='Whether to run the default test',
        default=False)
    return parser.parse_args()


def output_file(OUTPUT_FILE, cmd_args, run_time):
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'a') as fout:
            fout.write(
                f"{cmd_args.batch_size}\t{cmd_args.seq_len}\t{cmd_args.hidden_size}\t{cmd_args.block_size}\t"
                f"{run_time}\n")


if __name__ == '__main__':
    num_attention_heads = 1
    num_rand_blocks = 3

    cmd_args = parse_test_args()
    DEFAULT_TEST = cmd_args.default_test
    OUTPUT_FILE = cmd_args.output_file

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write(
                "batch size\tsequence length\thidden\tblock size\telapsed time(ms)\n"
            )

    if not DEFAULT_TEST:
        seq_len = cmd_args.seq_len
        batch_size = cmd_args.batch_size
        size_per_head = cmd_args.hidden_size
        block_size = cmd_args.block_size

        from_seq_length = seq_len
        from_block_size = block_size
        to_seq_length = seq_len
        to_block_size = block_size

        run_time = test_BigBird(
            batch_size=batch_size,
            size_per_head=size_per_head,
            from_seq_length=from_seq_length,
            from_block_size=from_block_size,
            to_seq_length=to_seq_length,
            to_block_size=to_block_size,
            num_rand_blocks=num_rand_blocks,
            num_attention_heads=num_attention_heads,
            output_file=OUTPUT_FILE)
    else:
        run_time = test_BigBird(
            batch_size=32,
            size_per_head=512,
            from_seq_length=512,
            from_block_size=64,
            to_seq_length=512,
            to_block_size=64,
            num_rand_blocks=num_rand_blocks,
            num_attention_heads=num_attention_heads,
            output_file=OUTPUT_FILE)
