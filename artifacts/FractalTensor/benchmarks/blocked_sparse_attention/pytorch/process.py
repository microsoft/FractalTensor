import os
import json
import glob
from tqdm import tqdm


def parse(input_file: str, output_file: str):
    data = json.load(open(input_file, encoding='utf-8'))
    trace_events = {}

    for event in data['traceEvents']:
        if 'cat' not in event or event['cat'] != 'Kernel':
            continue

        if event['name'] in trace_events:
            trace_events[event['name']]['dur'] += float(event['dur']) / 1000.
            trace_events[event['name']]['call_num'] += 1
        else:
            trace_events[event['name']] = {
                'dur': float(event['dur']) / 1000.,  # to ms
                'call_num': 1
            }

    sorted_events = sorted(
        trace_events.items(), key=lambda item: item[1]['dur'], reverse=True)

    with open(output_file, 'w') as fout, open('kernel_name.txt', 'w') as fn:
        fout.write('name\tdur\tcall_num\n')
        for k, v in sorted_events:
            fout.write('%s\t%f\t%d\n' % (k, v['dur'], v['call_num']))
            fn.write('%s\n' % k)


def stats(input_file: str, fout):
    compute_kernels = set()
    with open('figures/compute_kernels.tsv', 'r') as f:
        for line in f:
            compute_kernels.add(line.strip())

    compute_time = 0.
    non_compute_time = 0.
    with open(input_file, 'r') as fin:
        for i, line in enumerate(fin):
            if not i:
                continue
            name, dur, _ = line.strip().split('\t')
            if name in compute_kernels:
                compute_time += float(dur)
            else:
                non_compute_time += float(dur)
    test_name = os.path.splitext(os.path.split(input_file)[-1])[0]
    bs, seq, block = test_name.split('_')
    bs = bs.replace('bs', '')
    seq = seq.replace('seq', '')
    block = block.replace('block', '')
    fout.write('%s,%s\t%.5f\t%.5f\t%.5f\n' %
               (seq, block, compute_time, non_compute_time,
                compute_time + non_compute_time))


if __name__ == '__main__':
    dirbase = 'log'
    dirname = 'bs4_seq1024_block128'
    input_file_name = os.listdir(os.path.join(dirbase, dirname))[0]
    output_file = os.path.join(dirbase, dirname + '.tsv')
    parse(os.path.join(dirbase, dirname, input_file_name), output_file)

    with open('figures/bigbird_compute_vs_noncompute.tsv', 'w') as fout:
        fout.write('test name\tcompute\tnon-compute\ttotal\n')
        for log_file in glob.glob('log/*.tsv'):
            stats(log_file, fout)
