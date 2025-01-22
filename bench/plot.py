import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 定义一个 marker 迭代器


def style_iterator():
    markers = [
        ".", "v", "s", "|", "o", "^", "p", "_", ",", "<",
        "h", "+", ">", "H", "x", "D", "*", "d"
    ]
    linestyles = ['-', '--', '-.', ':']
    style_combinations = list(itertools.product(linestyles, markers))
    style_cycle = itertools.cycle(style_combinations)
    return style_cycle


def get_file_name(file_path):
    path = Path(file_path)
    file_name_without_extension = path.stem
    return file_name_without_extension


def list_files(directory, filename_filter: str):
    if not os.path.isdir(directory):
        print(f"error: directory '{directory}' not exists")
        return []
    all_files = os.listdir(directory)
    matched_files = [
        f"{directory}/{file}" for file in all_files if filename_filter in file and os.path.isfile(os.path.join(directory, file))]

    return matched_files


plt.style.use('ggplot')
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16


def clean_data(recall, qps, threshold=0.95):
    recall, qps = np.array(recall), np.array(qps)
    assert len(recall) == len(qps)
    not_dominated_flags = []
    i = 0
    for r, q in zip(recall, qps):
        j = 0
        is_dominated = False
        for r1, q1 in zip(recall, qps):
            if i != j and r1 >= r and q1 > threshold*q:
                is_dominated = True
            j += 1
        i += 1

        not_dominated_flags.append(not is_dominated)
    recall, qps = recall[not_dominated_flags], qps[not_dominated_flags]
    sorted_pairs = sorted(zip(recall, qps), key=lambda x: x[0])
    recall_sorted, qps_sorted = zip(*sorted_pairs)
    return recall_sorted, qps_sorted


def plot(ax, legend, recall, qps, **kwargs):
    recall, qps = clean_data(recall, qps)
    ax.plot(recall, qps, label=legend, **kwargs)
    ax.set_xlabel("Recall")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.grid(which="both", linestyle="--",
            linewidth=0.5, color="gray", alpha=0.7)


def plot_group(path_list, title, xlim=(0.9, 1), ylim=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    styles = style_iterator()
    for file_path in path_list:
        linestyle, marker = next(styles)
        df = pd.read_csv(file_path)
        legend = get_file_name(file_path)
        recall = df["recall"].values
        qps = df["QPS"].values
        plot(ax, legend, recall, qps, linestyle=linestyle, marker=marker)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f"{title}.png", bbox_inches="tight", dpi=400)


result_dir = "../results"

plot_group(list_files(result_dir, "nytimes"),
           "tmp_nytimes", ylim=(0, 3000))
plot_group(list_files(result_dir, "gist"),
           "tmp_gist", ylim=(0, 1500))
