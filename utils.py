import os
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