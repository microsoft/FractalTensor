#!/usr/bin/env python
import os

log_dir = "../logs"


def triton_rnn_fig2(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(22):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times.append(float(time))
    return times


def pt_rnn_fig2(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(22):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times.append(float(time))
    return times


def tf_rnn_fig2_auto(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(1):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times.append(float(time))
    return times


def tf_rnn_fig2_graph(file_name):
    times1, times2 = [], []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(1):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times1.append(float(time))

        for _ in range(1):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, time = line.split("\t")
            times2.append(float(time))

    return times1, times2


def cudnn_rnn_fig2_auto(file_name):
    times = []
    with open(os.path.join(log_dir, file_name), "r") as fin:
        for _ in range(22):
            fin.readline()

        for i in range(6):
            line = fin.readline().strip()
            _, _, _, _, time = line.split("\t")
            times.append(float(time))
    return times


def lstm_scale_with_fig2(fout):
    triton_times = triton_rnn_fig2("triton_stacked_lstm.tsv")
    pt_times = pt_rnn_fig2("pt_stacked_lstm.tsv")
    tf_autograph_times = tf_rnn_fig2_auto("tf2_stacked_lstm.tsv")
    tf_whileop_times, tf_graphmode_times = tf_rnn_fig2_graph(
        "tf1_stacked_lstm.tsv")
    cudnn_times = cudnn_rnn_fig2_auto("ft_stacked_lstm.tsv")

    res_pt_str = ""
    res_triton_str = ""
    res_tf1_str = ""
    res_tf2_str = ""
    res_tf3_str = ""
    res_cudnn_str = ""

    for i in range(6):
        res_pt_str += ("{:.4f}\t".format(pt_times[i]))
        res_triton_str += ("{:.4f}\t".format(triton_times[i]))
        res_tf1_str += ("{:.4f}\t".format(tf_autograph_times[i]))
        res_tf2_str += ("{:.4f}\t".format(tf_whileop_times[i]))
        res_tf3_str += ("{:.4f}\t".format(tf_graphmode_times[i]))
        res_cudnn_str += ("{:.4f}\t".format(cudnn_times[i]))

    fout.write("PT-JIT\t" + res_pt_str + "\t\n")
    fout.write("Triton\t" + res_triton_str + "\t\n")
    fout.write("TF-AutoGraph\t" + res_tf1_str + "\t\n")
    fout.write("TF-WhileOPLSTM\t" + res_tf2_str + "\t\n")
    fout.write("TF-GraphMode\t" + res_tf3_str + "\t\n")
    fout.write("CuDNN\t" + res_cudnn_str + "\t\n")


def post_processing_fig2():
    with open("figure2_rnn_scale_with_depths.tsv", "w") as fout:
        fout.write("Experiment Name\t\t1\t4\t8\t12\t16\t20\n")

        lstm_scale_with_fig2(fout)


if __name__ == "__main__":
    post_processing_fig2()
