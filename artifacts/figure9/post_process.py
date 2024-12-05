#!/usr/bin/env python
import os

log_dir = "../logs"


def triton_rnn_scale_with_depth(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(4):
            fin.readline()

        for i in range(12):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times.append(float(time))
    return times


def pt_rnn_scale_with_depth(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(4):
            fin.readline()

        for i in range(12):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times.append(float(time))
    return times


def triton_rnn_scale_with_length(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(16):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times.append(float(time))
    return times


def pt_rnn_scale_with_length(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(16):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times.append(float(time))
    return times


def lstm_scale_with_depth(fout):
    triton_times = triton_rnn_scale_with_depth("triton_stacked_lstm.tsv")
    pt_times = pt_rnn_scale_with_depth("pt_stacked_lstm.tsv")

    with open(os.path.join(log_dir, "ft_stacked_lstm.tsv"), "r") as fin:
        for _ in range(4):
            fin.readline()

        res_ours_str = ""
        res_cudnn_str = ""
        res_triton_str = ""
        res_pt_str = ""

        for i in range(5):
            line = fin.readline().strip()
            _, _, _, time, time_cudnn = line.split("\t")

            res_ours_str += (time + "\t")
            res_cudnn_str += (time_cudnn + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[i]))

        line = fin.readline().strip()
        _, _, _, time, time_cudnn = line.split("\t")
        res_ours_str += (time + "\n")
        res_cudnn_str += (time_cudnn + "\n")

        res_triton_str += ("{:.4f}\n".format(triton_times[5]))
        res_pt_str += ("{:.4f}\n".format(pt_times[5]))

        fout.write("lstm_ft_256\t" + res_ours_str)
        fout.write("lstm_cudnn_256\t" + res_cudnn_str)
        fout.write("lstm_triton_256\t" + res_triton_str)
        fout.write("lstm_PyTorch_256\t" + res_pt_str)

        res_ours_str = ""
        res_cudnn_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(5):
            line = fin.readline().strip()
            _, _, _, time, cudnn_time = line.split("\t")

            res_ours_str += (time + "\t")
            res_cudnn_str += (cudnn_time + "\t")

            res_triton_str += ("{:.4f}\t".format(triton_times[6 + i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[6 + i]))

        line = fin.readline().strip()
        _, _, _, time, cudnn_time = line.split("\t")

        res_ours_str += (time + "\n")
        res_cudnn_str += (time_cudnn + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[11]))
        res_pt_str += ("{:.4f}\n".format(pt_times[11]))

        fout.write("lstm_ft_1024\t" + res_ours_str)
        fout.write("lstm_cudnn_1024\t" + res_cudnn_str)
        fout.write("lstm_triton_1024\t" + res_triton_str)
        fout.write("lstm_PyTorch_1024\t" + res_pt_str)


def dilate_scale_with_depth(fout):
    triton_times = triton_rnn_scale_with_depth("triton_dilated_lstm.tsv")
    pt_times = pt_rnn_scale_with_depth("pt_dilated_lstm.tsv")

    with open(os.path.join(log_dir, "ft_dilated_lstm.tsv"), "r") as fin:
        for _ in range(4):
            fin.readline()

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(5):
            line = fin.readline().strip()
            _, _, _, time, _, _ = line.split("\t")

            res_str += (time + "\t")

            res_triton_str += ("{:.4f}\t".format(triton_times[i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[i]))

        line = fin.readline().strip()
        _, _, _, time, _, _ = line.split("\t")

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[5]))
        res_pt_str += ("{:.4f}\n".format(pt_times[5]))

        fout.write("dilated_lstm_ft_256\t" + res_str)
        fout.write("dilated_lstm_triton_256\t" + res_triton_str)
        fout.write("dilated_lstm_PyTorch_256\t" + res_pt_str)

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(5):
            line = fin.readline().strip()
            _, _, _, time, _, _ = line.split("\t")

            res_triton_str += ("{:.4f}\t".format(triton_times[6 + i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[6 + i]))

            res_str += (time + "\t")

        line = fin.readline().strip()
        _, _, _, time, _, _ = line.split("\t")

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[11]))
        res_pt_str += ("{:.4f}\n".format(pt_times[11]))

        fout.write("dilated_lstm_ft_1024\t" + res_str)
        fout.write("dilated_lstm_triton_1024\t" + res_triton_str)
        fout.write("dilated_lstm_PyTorch_1024\t" + res_pt_str)


def grid_scale_with_depth(fout):
    triton_times = triton_rnn_scale_with_depth("triton_grid_lstm.tsv")
    pt_times = pt_rnn_scale_with_depth("pt_grid_lstm.tsv")

    with open(os.path.join(log_dir, "ft_grid_lstm.tsv"), "r") as fin:
        for _ in range(4):
            fin.readline()

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(5):
            line = fin.readline().strip()
            line_splits = line.split("\t")
            time = line_splits[-1]

            res_str += (time + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[i]))

        line = fin.readline().strip()
        line_splits = line.split("\t")
        time = line_splits[-1]

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[5]))
        res_pt_str += ("{:.4f}\n".format(pt_times[5]))

        fout.write("grid_rnn_ft_256\t" + res_str)
        fout.write("grid_rnn_triton_256\t" + res_triton_str)
        fout.write("grid_rnn_PyTorch_256\t" + res_pt_str)

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(5):
            line = fin.readline().strip()
            line_splits = line.split("\t")
            time = line_splits[-1]

            res_str += (time + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[6 + i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[6 + i]))

        line = fin.readline().strip()
        line_splits = line.split("\t")
        time = line_splits[-1]

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[11]))
        res_pt_str += ("{:.4f}\n".format(pt_times[11]))

        fout.write("grid_rnn_ft_1024\t" + res_str)
        fout.write("grid_rnn_triton_1024\t" + res_triton_str)
        fout.write("grid_rnn_PyTorch_1024\t" + res_pt_str)


def lstm_scale_with_length(fout):
    triton_times = triton_rnn_scale_with_length("triton_stacked_lstm.tsv")
    pt_times = pt_rnn_scale_with_length("pt_stacked_lstm.tsv")

    with open(os.path.join(log_dir, "ft_stacked_lstm.tsv"), "r") as fin:
        for i in range(16):
            fin.readline()

        res_ours_str = ""
        res_cudnn_str = ""
        res_triton_str = ""
        res_pt_str = ""

        for i in range(2):
            line = fin.readline()
            _, _, _, time, cudnn = line.strip().split("\t")

            res_ours_str += (time + "\t")
            res_cudnn_str += (cudnn + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[i]))

        line = fin.readline()
        _, _, _, time, cudnn = line.strip().split("\t")

        res_ours_str += (time + "\n")
        res_cudnn_str += (cudnn + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[2]))
        res_pt_str += ("{:.4f}\n".format(pt_times[2]))

        fout.write("lstm_ft_256\t" + res_ours_str)
        fout.write("lstm_cudnn_256\t" + res_cudnn_str)
        fout.write("lstm_triton_256\t" + res_triton_str)
        fout.write("lstm_PyTorch_256\t" + res_pt_str)

        res_ours_str = ""
        res_cudnn_str = ""
        res_triton_str = ""
        res_pt_str = ""

        for j in range(2):
            line = fin.readline()
            _, _, _, time, cudnn = line.strip().split("\t")

            res_ours_str += (time + "\t")
            res_cudnn_str += (cudnn + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[3 + i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[3 + i]))

        line = fin.readline()
        _, _, _, time, cudnn = line.strip().split("\t")

        res_ours_str += (time + "\n")
        res_cudnn_str += (cudnn + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[5]))
        res_pt_str += ("{:.4f}\n".format(pt_times[5]))

        fout.write("lstm_ft_1024\t" + res_ours_str)
        fout.write("lstm_cudnn_1024\t" + res_cudnn_str)
        fout.write("lstm_triton_1024\t" + res_triton_str)
        fout.write("lstm_PyTorch_1024\t" + res_pt_str)


def dilate_scale_with_length(fout):
    triton_times = triton_rnn_scale_with_length("triton_dilated_lstm.tsv")
    pt_times = pt_rnn_scale_with_length("pt_dilated_lstm.tsv")

    with open(os.path.join(log_dir, "ft_dilated_lstm.tsv"), "r") as fin:
        for _ in range(16):
            fin.readline()

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(2):
            line = fin.readline()

            line_splits = line.strip().split("\t")
            time = line_splits[3]

            res_str += (time + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[i]))

        line = fin.readline()
        line_splits = line.strip().split("\t")
        time = line_splits[3]

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[2]))
        res_pt_str += ("{:.4f}\n".format(pt_times[2]))

        fout.write("dilated_ft_256\t" + res_str)
        fout.write("dilated_triton_256\t" + res_triton_str)
        fout.write("dilated_PyTorch_256\t" + res_pt_str)

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(2):
            line = fin.readline()

            line_splits = line.strip().split("\t")
            time = line_splits[3]

            res_str += (time + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[3 + i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[3 + i]))

        line = fin.readline()
        line_splits = line.strip().split("\t")
        time = line_splits[3]

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[5]))
        res_pt_str += ("{:.4f}\n".format(pt_times[5]))

        fout.write("dilated_ft_1024\t" + res_str)
        fout.write("dilated_triton_1024\t" + res_triton_str)
        fout.write("dilated_PyTorch_1024\t" + res_pt_str)


def grid_scale_with_length(fout):
    triton_times = triton_rnn_scale_with_length("triton_grid_lstm.tsv")
    pt_times = pt_rnn_scale_with_length("pt_grid_lstm.tsv")

    with open(os.path.join(log_dir, "ft_grid_lstm.tsv"), "r") as fin:
        for _ in range(16):
            fin.readline()

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(2):
            line = fin.readline()

            line_splits = line.strip().split("\t")
            time = line_splits[-1]

            res_str += (time + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[i]))

        line = fin.readline()
        line_splits = line.strip().split("\t")
        time = line_splits[-1]

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[2]))
        res_pt_str += ("{:.4f}\n".format(pt_times[2]))

        fout.write("grid_rnn_ft_256\t" + res_str)
        fout.write("grid_rnn_triton_256\t" + res_triton_str)
        fout.write("grid_rnn_PyTorch_256\t" + res_pt_str)

        res_str = ""
        res_triton_str = ""
        res_pt_str = ""
        for i in range(2):
            line = fin.readline()

            line_splits = line.strip().split("\t")
            time = line_splits[-1]

            res_str += (time + "\t")
            res_triton_str += ("{:.4f}\t".format(triton_times[3 + i]))
            res_pt_str += ("{:.4f}\t".format(pt_times[3 + i]))

        line = fin.readline()
        line_splits = line.strip().split("\t")
        time = line_splits[-1]

        res_str += (time + "\n")
        res_triton_str += ("{:.4f}\n".format(triton_times[5]))
        res_pt_str += ("{:.4f}\n".format(pt_times[5]))

        fout.write("grid_rnn_ft_1024\t" + res_str)
        fout.write("grid_rnn_triton_1024\t" + res_triton_str)
        fout.write("grid_rnn_PyTorch_1024\t" + res_pt_str)


def post_processing_vary_depth():
    with open("figure9_rnn_scale_with_depths.tsv", "w") as fout:
        fout.write("Experiment Name\t1\t2\t4\t8\t16\t32\n")

        lstm_scale_with_depth(fout)
        dilate_scale_with_depth(fout)
        grid_scale_with_depth(fout)


def post_processing_vary_length():
    with open("figure9_rnn_scale_with_lengths.tsv", "w") as fout:
        fout.write("Experiment Name\t32\t64\t128\n")

        lstm_scale_with_length(fout)
        dilate_scale_with_length(fout)
        grid_scale_with_length(fout)


if __name__ == "__main__":
    post_processing_vary_depth()
    post_processing_vary_length()
