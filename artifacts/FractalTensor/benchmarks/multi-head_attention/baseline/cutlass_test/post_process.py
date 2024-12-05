def post_process():
    elapsed_times = []
    # with open('cutlass_a6000.txt', 'r') as flog:
    with open('cutlass_a100.txt', 'r') as flog:
        for line in flog:
            if not 'Runtime: ' in line:
                continue

            line_splits = line.split()
            elapsed_times.append(float(line_splits[1]))

    # with open('../figures/MHA_A6000.tsv', 'r') as fin:
    with open('../figures/MHA_A100.tsv', 'r') as fin:
        data = fin.readlines()

    assert len(elapsed_times) == len(data) - 1

    header = data[0]
    with open('../figures/MHA_A100-2.tsv', 'w') as fout:
        fout.write('%s\tcutlass(ms)\n' % (header.strip()))

        for i in range(1, len(data)):
            fout.write('%s\t%.4f\n' % (data[i].strip(), elapsed_times[i - 1]))


if __name__ == '__main__':
    post_process()
