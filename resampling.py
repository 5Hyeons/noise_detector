import os
import librosa
import numpy as np

from tqdm import tqdm
from scipy.io import wavfile
from functools import partial
from multiprocessing import Pool, cpu_count

def process(wav_path, sr):
    '''
    References
    ----------
    https://github.com/OlaWod/FreeVC/blob/main/downsample.py
    '''
    wav, sr = librosa.load(wav_path, sr=sr)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = 0.98 * wav / peak
    wavfile.write(
        wav_path,
        sr,
        (wav * np.iinfo(np.int16).max).astype(np.int16)
    )

def resampling(root, sr) -> None:
    '''
    This function is a resampling function using multoprocessing.
    and it can find wav files reculsively.

    Parameters
    ---
    root : Root directory containing wav files
    sr : The sampling rate to transform
    '''
    pool = Pool(processes=cpu_count()-2)

    wav_path = []
    def find_wav(path):
        if '.wav' in path:
            wav_path.append(path)
            return
        if os.path.isdir(path):
            for lst in os.listdir(path):
                find_wav(os.path.join(path, lst))
    find_wav(root)

    f = partial(process, sr=sr)
    for _ in tqdm(pool.imap_unordered(f, wav_path)):
        pass
