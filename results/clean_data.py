import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_file_name(file_path):
    path = Path(file_path)
    file_name_without_extension = path.stem
    return file_name_without_extension


def list_csv(directory, require_keywords=[], exclude_keywords=[]):
    if not os.path.isdir(directory):
        print(f"error: directory '{directory}' not exists")
        return []
    all_files = os.listdir(directory)
    matched_files = []
    for file in all_files:
        if not file.endswith("csv"):
            continue
        if os.path.isfile(os.path.join(directory, file)):
            qualified = True
            for name in require_keywords:
                if name not in file:
                    qualified = False
                    break
            for name in exclude_keywords:
                if name in file:
                    qualified = False
                    break
            if qualified:
                matched_files.append(f"{directory}/{file}")
    return matched_files


def clean_data(recall, qps):
    recall, qps = np.array(recall), np.array(qps)
    assert len(recall) == len(qps)
    not_dominated_flags = []
    i = 0
    for r, q in zip(recall, qps):
        j = 0
        is_dominated = False
        if r < 0.9:
            is_dominated = True
        else:
            for r1, q1 in zip(recall, qps):
                if i != j and (r1 >= r) and q1 > q:
                    is_dominated = True
                j += 1
            i += 1

        not_dominated_flags.append(not is_dominated)
    return not_dominated_flags


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

target_dir = f"{script_dir}/cleand"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

path_list = list_csv(script_dir, [], ["cleaned"])
for file_path in path_list:
    df = pd.read_csv(file_path)
    mask = clean_data(df["recall"].values, df["QPS"].values)
    cleaned_df = df[mask]
    cleaned_df = cleaned_df.sort_values(by="recall")
    cleaned_df.to_csv(
        f"{target_dir}/cleaned_{get_file_name(file_path)}.csv", index=False)
