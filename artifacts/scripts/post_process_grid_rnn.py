def post_process(file_name):
    with open(file_name, "r") as fin, \
            open("logs/pt_grid_lstm.tsv", "w") as fout:
        fout.write(
            "depth\t[seq_length, batch_size, hidden_size]\tPyTorch(ms)\n")

        for line in fin:
            if not "bench-grid" in line:
                continue
            line_splits = line.strip().split("\t")
            depth = line_splits[2]
            length = line_splits[4]
            batch_size = line_splits[6]
            hidden_size = line_splits[8]
            time = line_splits[-1]
            fout.write(
                f"{depth}\t[{length}, {batch_size}, {hidden_size}]\t{time}\n")


if __name__ == "__main__":
    post_process("logs/pt_grid_lstm.log")
