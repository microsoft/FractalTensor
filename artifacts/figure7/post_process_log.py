#!/usr/bin/env python

import os

log_dir = "../logs"


def post_process_cutlass(file_name):
    elapsed_times = []
    with open(file_name, 'r') as flog:
        for line in flog:
            if not 'Runtime: ' in line:
                continue

            line_splits = line.split()
            elapsed_times.append(float(line_splits[1]))

    return elapsed_times[1:]


def post_process_baseline_lstm(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as f:
        f.readline()  # skip header

        for i in range(3):
            line = f.readline()
            _, problem_shape, time = line.strip().split("\t")

            problem_shape = problem_shape.replace("[", "")
            problem_shape = problem_shape.replace("]", "")

            hidden = problem_shape.split(",")[-1]
            times.append([hidden, float(time)])
    return times


def post_process_lstm_log():
    lstm_records = []
    with open(os.path.join(log_dir, "ft_stacked_lstm.tsv"), "r") as f:
        f.readline()  # skip header

        for i, line in enumerate(f):
            if i > 2:
                break
            _, problem_shape, _, time, cudnn_time = line.strip().split("\t")
            problem_shape = problem_shape.replace("[", "")
            problem_shape = problem_shape.replace("]", "")

            length, batch, hidden = problem_shape.split(",")
            lstm_records.append([hidden, time, cudnn_time])

    pt_times = post_process_baseline_lstm("pt_stacked_lstm.tsv")
    triton_times = post_process_baseline_lstm("triton_stacked_lstm.tsv")

    with open("0_figure7_lstm.tsv", "w") as f:
        f.write("Experiment Name\thidden 256\thidden 512\thidden 1024\n")

        f.write("ft(ms)\t{}\t{}\t{}\n".format(
            lstm_records[0][1], lstm_records[1][1], lstm_records[2][1]))
        f.write("cudnn(ms)\t{}\t{}\t{}\n".format(
            lstm_records[0][2], lstm_records[1][2], lstm_records[2][2]))
        f.write("pt(ms)\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            pt_times[0][1], pt_times[1][1], pt_times[2][1]))
        f.write("triton(ms)\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            triton_times[0][1], triton_times[1][1], triton_times[2][1]))


def post_process_dilated_log():
    dilated_records = []
    with open(os.path.join(log_dir, "ft_dilated_lstm.tsv"), "r") as f:
        f.readline()  # skip header

        for i, line in enumerate(f):
            if i > 2:
                break
            _, problem_shape, _, time, _, _ = line.strip().split("\t")
            problem_shape = problem_shape.replace("[", "")
            problem_shape = problem_shape.replace("]", "")

            _, _, hidden = problem_shape.split(",")
            dilated_records.append([hidden, time])

    pt_times = post_process_baseline_lstm("pt_dilated_lstm.tsv")
    triton_times = post_process_baseline_lstm("triton_dilated_lstm.tsv")

    with open("1_figure7_dilated.tsv", "w") as f:
        f.write("Experiment Name\thidden 256\thidden 512\thidden 1024\n")
        f.write("ft dilated(ms)\t{}\t{}\t{}\n".format(dilated_records[0][1],
                                                      dilated_records[1][1],
                                                      dilated_records[2][1]))
        f.write("pt(ms)\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            pt_times[0][1], pt_times[1][1], pt_times[2][1]))
        f.write("triton(ms)\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            triton_times[0][1], triton_times[1][1], triton_times[2][1]))


def post_process_grid_log():
    grid_records = []
    with open(os.path.join(log_dir, "ft_grid_lstm.tsv"), "r") as f:
        f.readline()  # skip header

        for i, line in enumerate(f):
            if i > 2:
                break
            _, problem_shape, _, _, _, _, time = line.strip().split("\t")
            problem_shape = problem_shape.replace("[", "")
            problem_shape = problem_shape.replace("]", "")

            _, hidden, _, _, _ = problem_shape.split(",")
            grid_records.append([hidden, time])

    pt_times = post_process_baseline_lstm("pt_grid_lstm.tsv")
    triton_times = post_process_baseline_lstm("triton_grid_lstm.tsv")

    with open("2_figure7_grid.tsv", "w") as f:
        f.write("Experiment Name\thidden 256\thidden 512\thidden 1024\n")
        f.write("ft grid(ms)\t{}\t{}\t{}\n".format(
            grid_records[0][1], grid_records[1][1], grid_records[2][1]))

        f.write("pt(ms)\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            pt_times[0][1], pt_times[1][1], pt_times[2][1]))
        f.write("triton(ms)\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            triton_times[0][1], triton_times[1][1], triton_times[2][1]))


def post_process_triton_back2back():

    times = []
    with open(os.path.join(log_dir, "triton_b2b_gemm.tsv"), "r") as fin:
        fin.readline()

        for line in fin:
            time_str = line.strip().split("\t")[-1]

            time_str = time_str.replace("ms", "")
            time = time_str.split(":")[-1]

            times.append(float(time))
    return times


def post_process_back2back_gemm():
    with open(os.path.join(log_dir, "ft_b2b_gemm.tsv"), "r") as fin, \
            open("3_figure7_b2b_gemm.tsv", "w") as fout:
        fin.readline()  # skip header
        fout.write("Experiment Name\tft (ms)\tcublas (ms)\ttriton (ms)\n")

        times = post_process_triton_back2back()

        for i, line in enumerate(fin):
            line_splits = line.strip().split("\t")
            problem_shape = line_splits[0]

            problem_shape = problem_shape.replace("[", "")
            problem_shape = problem_shape.replace("]", ",")
            shapes = problem_shape.split(",")
            m = shapes[0]
            n = ''.join(shapes[3].split())

            cublas = line_splits[5]
            time = line_splits[6]

            exp_name = f"M={m}, N={n}"
            fout.write(f"{exp_name}\t{time}\t{cublas}\t{times[i]:.4f}\n")


def post_process_cutlass(file_name):
    times = []
    with open(file_name, 'r') as flog:
        for line in flog:
            if not 'Runtime: ' in line:
                continue

            line_splits = line.split()
            times.append(float(line_splits[1]))
    return times


def post_process_fa(file_name):
    times = []
    with open(file_name, 'r') as flog:
        flog.readline()
        for line in flog:
            line = line.strip()
            if not line:
                continue
            time = line.strip().split("\t")[-1]
            times.append(float(time))
    return times


def post_process_triton_attention(file_name):
    times = []
    with open(file_name, 'r') as flog:
        flog.readline()
        for line in flog:
            line = line.strip()
            if not line:
                continue
            time = line.strip().split("\t")[-1]
            times.append(float(time))
    return times


def post_process_attention_log():
    with open(os.path.join(log_dir, "ft_attention.tsv"), "r") as fin:
        times_128 = post_process_cutlass(
            os.path.join(log_dir, 'cutlass_attn_128.log'))

        flash_attn_times = post_process_fa(
            os.path.join(log_dir, 'pt_attention.tsv'))

        triton_times = post_process_triton_attention(
            os.path.join(log_dir, 'triton_attention.tsv'))

        if len(times_128) != 8:
            raise ValueError("Error: cutlass_attn_128.log is not complete")

        with open("4_figure7_attention.tsv", "w") as fout:
            fout.write("length\thidden\tft(ms)\t"
                       "flash attention(ms)\ttriton\tcutlass(ms)\n")

            for i in range(8):
                line = fin.readline()
                length, hidden, time = line.strip().split("\t")
                fout.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{}\n".format(
                    length, hidden, time, flash_attn_times[i], triton_times[i],
                    times_128[i]))

        times_256 = post_process_cutlass('../logs/cutlass_attn_256.log')
        if len(times_256) != 8:
            raise ValueError("Error: cutlass_attn_256.log is not complete")

        with open("5_figure7_attention.tsv", "w") as fout:
            fout.write("length\thidden\tft(ms)\t"
                       "flash attention(ms)\ttriton\tcutlass(ms)\n")

            for i in range(8):
                line = fin.readline()
                length, hidden, time = line.strip().split("\t")
                fout.write("{}\t{}\t{}\t{:.4f}\t{}\t{}\n".format(
                    length,
                    hidden,
                    time,
                    flash_attn_times[8 + i],
                    -1,
                    times_256[i],
                ))


def post_process_pt_bigbird(file_name):
    with open(file_name, 'r') as flog:
        for i in range(3):
            flog.readline()
        time_str = flog.readline().split(",")[-1].replace("ms", "")
        time = "".join(time_str.split(":")[-1].split())
    return time


def post_process_triton_bigbird(file_name):
    times = []
    with open(file_name, 'r') as flog:
        for line in flog:
            time = line.strip().split("\t")[-1]
            times.append(float(time))
    return times


def post_process_bigbird_log():
    pt_time = post_process_pt_bigbird(os.path.join(log_dir, "pt_bigbird.tsv"))
    triton_times = post_process_triton_bigbird(
        os.path.join(log_dir, "triton_bigbird.tsv"))

    with open(os.path.join(log_dir, "ft_bigbird.tsv"), "r") as fin, \
            open("6_figure7_bigbird.tsv", "w") as fout:
        fin.readline()  # skip header
        fout.write("length\tft (ms)\ttriton(ms)\tPyTorch(ms)\n")
        for i, line in enumerate(fin):
            _, length, _, _, time = line.strip().split("\t")
            if i == 0:
                fout.write("{}\t{}\t{:.4f}\t{}\n".format(
                    length, time, triton_times[i], pt_time))
            else:
                fout.write("{}\t{}\t{:.4f}\t-1\n".format(
                    length, time, triton_times[i]))


if __name__ == "__main__":
    post_process_lstm_log()
    post_process_dilated_log()
    post_process_grid_log()
    post_process_back2back_gemm()
    post_process_attention_log()
    post_process_bigbird_log()
