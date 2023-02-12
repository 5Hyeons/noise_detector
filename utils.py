import os
import glob
import pandas as pd

from typing import Tuple, List

def get_fname_with_label(csv_path:str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    cleans = df[df['state']=='Clean']['fname'].to_list()
    noises = df[df['state']=='Noisy']['fname'].to_list()

    return cleans, noises

def get_label_value(csv_path):
    df = pd.read_csv(csv_path)
    values = df['state'].values
    sections = ['Noisy', 'Clean', 'Unknown']
    ys = [0, 0, 0]
    for value in values:
        for i, section in enumerate(sections):
            if value == section:
                ys[i] += 1
                break
    return sections, ys

def get_state_ratio(csv_dir):
    state_counts = {'Clean':[], 'Noisy':[], 'Unknown':[]}

    csv_paths = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        vc = df['state'].value_counts()
        for state in state_counts.keys():
            try:
                state_counts[state].append(vc[state]/len(df))
            except KeyError:
                state_counts[state].append(0)
    return state_counts

def apply_differential(ys, order=0):
    for _ in range(order):
        ys = [y2-y1 for y1, y2 in zip(ys, ys[1:])]

    return ys

            