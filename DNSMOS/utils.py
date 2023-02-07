import os
import pandas as pd

from typing import Tuple, List

def csv_to_dict(csv_path:str) -> dict:
    df = pd.read_csv(csv_path)
    df.set_index('fname', inplace=True)
    return df.to_dict()

def dict_to_csv(dic:dict, csv_path:str) -> None:
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(csv_path, index_label='fname')

def get_fnames_with_MOS(csv_path:str, score:int) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    upper = df[df['P808_MOS']>=score]['fname']
    lower = df[df['P808_MOS']<score]['fname']

    return upper.to_list(), lower.to_list()
