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

# 특정 MOS 점수를 기준으로 상, 하로 나누어서 출력
def get_fnames_with_MOS(csv_path:str, score:int) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    upper = df[df['P808_MOS']>=score]
    upper_fname = upper['fname'].to_list()
    upper_mos = upper['P808_MOS'].to_list()
    lower = df[df['P808_MOS']<score]
    lower_fname = lower['fname'].to_list()
    lower_mos = lower['P808_MOS'].to_list()

    return upper_fname, upper_mos, lower_fname, lower_mos

# step을 기준으로 MOS 점수를 구간별로 나누어 구간끼리 통합하여 출력
def get_label_value(csv_path, column, step):
    df = pd.read_csv(csv_path)
    values = df[column].values
    # MOS의 최댓값이 5이므로 5를 step으로 나눔
    sections = [round(i*step, 3) for i in range(int(5//step))]
    sections.append(5)
    ys = [0] * len(sections)
    for value in values:
        for i, section in enumerate(sections[1:]):
            if value < section:
                ys[i] += 1
                break
    return sections, ys