import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.predictor import Predictor
from src.audio import read_as_melspectrogram
from src.transforms import get_transforms
from src.metrics import LwlrapBase
from src.utils import get_best_model_path, gmean_preds_blend
from src.datasets import get_data
from src import config

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--data_path', required=True, type=str, default=None)
parser.add_argument('--save_path', required=False, type=str, default=None)
args = parser.parse_args()

WORK_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
EXPERIMENT_DIR = WORK_DIR / 'data' / 'experiments' / args.experiment
PREDICTION_DIR = WORK_DIR / 'data'/ 'predictions' / args.experiment / 'probs'
#
DEVICE = 'cuda'
CROP_SIZE = 256
BATCH_SIZE = 16

def pred_fold(predictor, fold, input_data):
    fold_prediction_dir = PREDICTION_DIR
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


def blend_predictions(save_path):
    probs_df_lst = []
    for fold in config.folds:
        fold_probs_path = PREDICTION_DIR / f'probs_fold_{fold}.csv'
        probs_df = pd.read_csv(fold_probs_path)
        probs_df.set_index('fname', inplace=True)
        probs_df_lst.append(probs_df)

    blend_df = gmean_preds_blend(probs_df_lst)
    blend_df.to_csv(save_path)

def get_result(probs, save_path):
    probs_df = pd.read_csv(probs)
    probs_df.set_index('fname', inplace=True)
    labels = []
    for idx, row in probs_df.iterrows():
        if row['Speech'] > 0.8:
            if row['Noisy'] > 0.5:
                labels.append('Noisy')
            else:
                labels.append('Speech')
        else:
            labels.append('Unknown')
    result = pd.DataFrame(data=labels,
                          index=probs_df.index,
                          columns=['state'])
    result.index.name = 'fname'
    result.to_csv(save_path)

def prediction():
    transforms = get_transforms(False, CROP_SIZE)
    input_data = get_data(args.data_path)

    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = EXPERIMENT_DIR / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = Predictor(model_path, transforms,
                              BATCH_SIZE,
                              (config.audio.n_mels, CROP_SIZE),
                              (config.audio.n_mels, CROP_SIZE//4),
                              device=DEVICE)

        pred_fold(predictor, fold, input_data)

    print("Blend folds predictions")
    blend_probs_path = PREDICTION_DIR / 'probs.csv'
    blend_predictions(blend_probs_path)
    # save_path 입력하지 않을 경우, 데이터 폴더 명으로 저장.
    if args.save_path is None:
        args.save_path = str(args.data_path).rstrip('/')+'.csv'
        print(args.save_path)
    get_result(blend_probs_path, args.save_path)
    os.chmod(args.save_path, 0o0777)