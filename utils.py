import os
import pandas as pd

from typing import Tuple, List

def get_fname_with_label(csv_path:str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    cleans = df[df['state']=='Clean']['fname'].to_list()
    noises = df[df['state']=='Noisy']['fname'].to_list()

    return cleans, noises
