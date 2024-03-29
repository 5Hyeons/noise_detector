import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .src.predictor import Predictor
from .src.audio import read_as_melspectrogram
from .src.transforms import get_transforms
from .src.metrics import LwlrapBase
from .src.utils import get_best_model_path, gmean_preds_blend
from .src.datasets import get_data
from .src import config

WORK_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
EXPERIMENT_DIR = WORK_DIR / 'data' / 'experiments' 
PREDICTION_DIR = WORK_DIR / 'data'/ 'predictions'
#
CROP_SIZE = 256

def pred_fold(predictor, experiment, input_data, fold):
    fold_prediction_dir = PREDICTION_DIR / experiment / 'probs'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)
    fold_probs_path = fold_prediction_dir / f'probs_fold_{fold}.csv'

    fname_lst, images_lst, fpath_lst = input_data
    pred_lst = []
    for fname, image in zip(fname_lst, images_lst):
        pred = predictor.predict(image)
        pred = pred.mean(axis=0)
        pred_lst.append(pred)

    preds = np.stack(pred_lst, axis=0)
    subm_df = pd.DataFrame(data=preds,
                           index=fname_lst,
                           columns=config.classes)
    subm_df.index.name = 'fname'
    subm_df.to_csv(fold_probs_path)


def blend_predictions(experiment, output_path):
    probs_df_lst = []
    for fold in config.folds:
        fold_probs_path = PREDICTION_DIR / experiment / 'probs' / f'probs_fold_{fold}.csv'
        probs_df = pd.read_csv(fold_probs_path)
        probs_df.set_index('fname', inplace=True)
        probs_df_lst.append(probs_df)

    blend_df = gmean_preds_blend(probs_df_lst)
    blend_df.to_csv(output_path)

def get_result(probs):
    probs_df = pd.read_csv(probs)
    probs_df.set_index('fname', inplace=True)
    labels = []
    for idx, row in probs_df.iterrows():
        # labeling
        if row['Speech'] > 0.8:
            for col in probs_df.columns:
                if col == 'Speech': continue
                if row[col] > 0.5:
                    print(col)
                    labels.append('Noisy')
                    break
            else:
                labels.append('Clean')
        else:
            labels.append('Unknown')
    result = pd.DataFrame(data=labels,
                          index=probs_df.index,
                          columns=['state'])
    result.index.name = 'fname'
    return result

def predict(data_path, output_path=None, experiment='vctk_001', batch_size=16, device='cuda'):
    '''
    data_path: input source data dir.
    output_path: path to save output csv file. if None, the output will be saved as the input folder name.
    experiment: model name. (default: vctk_001)
    '''
    transforms = get_transforms(False, CROP_SIZE)
    input_data = get_data(data_path)

    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = EXPERIMENT_DIR / experiment / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = Predictor(model_path, transforms,
                              batch_size,
                              (config.audio.n_mels, CROP_SIZE),
                              (config.audio.n_mels, CROP_SIZE//4),
                              device=device)

        pred_fold(predictor, experiment, input_data, fold)

    print("Blend folds predictions")
    blend_probs_path = PREDICTION_DIR / experiment / 'probs'/ 'probs.csv'
    blend_predictions(experiment, blend_probs_path)

    # output_path 입력하지 않을 경우, 데이터 폴더 명으로 저장.
    if output_path is None:
        output_path = str(data_path).rstrip('/')+'.csv'

    labeling_df = get_result(blend_probs_path)
    labeling_df.sort_index(inplace=True)
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_df.set_index('fname', inplace=True)
        # 퀄리티 평가한 파일이 이미 있는 경우 concat
        if len(existing_df.index) == len(labeling_df.index) and all(existing_df.index == labeling_df.index):
            if 'state' not in existing_df.columns:
                labeling_df = pd.concat([labeling_df, existing_df], axis=1)
            elif 'P808_MOS' in existing_df.columns:
                for col in existing_df.columns:
                    if col == 'state': continue
                    labeling_df[col] = existing_df[col]
    
    labeling_df.to_csv(output_path)
    os.chmod(output_path, 0o0777)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str, default=None)
    parser.add_argument('--output_path', required=False, type=str, default=None)
    args = parser.parse_args()

    predict(args.data_path, args.output_path, args.experiment)