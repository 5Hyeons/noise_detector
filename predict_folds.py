import json
import argparse
import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.audio import read_as_melspectrogram
from src.transforms import get_transforms
from src.metrics import LwlrapBase
from src.utils import get_best_model_path, gmean_preds_blend
from src.datasets import get_test_data
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()


EXPERIMENT_DIR = config.experiments_dir / args.experiment
PREDICTION_DIR = config.predictions_dir / args.experiment
DEVICE = 'cuda'
CROP_SIZE = 256
BATCH_SIZE = 16

def pred_test_fold(predictor, fold, test_data):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'test'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    fname_lst, images_lst = test_data
    pred_lst = []
    for fname, image in zip(fname_lst, images_lst):
        pred = predictor.predict(image)

        pred_path = fold_prediction_dir / f'{fname}.npy'
        np.save(pred_path, pred)

        pred = pred.mean(axis=0)
        pred_lst.append(pred)

    preds = np.stack(pred_lst, axis=0)
    subm_df = pd.DataFrame(data=preds,
                           index=fname_lst,
                           columns=config.classes)
    subm_df.index.name = 'fname'
    subm_df.to_csv(fold_prediction_dir / 'probs.csv')


def blend_test_predictions():
    probs_df_lst = []
    for fold in config.folds:
        fold_probs_path = PREDICTION_DIR / f'fold_{fold}' / 'test' / 'probs.csv'
        probs_df = pd.read_csv(fold_probs_path)
        probs_df.set_index('fname', inplace=True)
        probs_df_lst.append(probs_df)

    blend_df = gmean_preds_blend(probs_df_lst)
    for idx, row in blend_df.iterrows():
        for idx2, value in zip(row.index, row.values):
            if 'speech' in idx2 or value < 0.3:
                continue
            print(idx, idx2, value)
        # break

    if config.kernel:
        blend_df.to_csv('submission.csv')
    else:
        blend_df.to_csv(PREDICTION_DIR / 'probs.csv')

if __name__ == "__main__":
    transforms = get_transforms(False, CROP_SIZE)
    test_data = get_test_data()

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

        print("Test predict")
        pred_test_fold(predictor, fold, test_data)

    print("Blend folds predictions")
    blend_test_predictions()
