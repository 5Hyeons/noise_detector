# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

import os
import glob
import argparse
import librosa
import concurrent.futures
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import soundfile as sf
import onnxruntime as ort

from tqdm import tqdm
from pathlib import Path
from requests import session

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
WORK_DIR = os.path.dirname(os.path.realpath(__file__))

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path, only_mos) -> None:
        self.only_mos = only_mos
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        if not self.only_mos:
            self.onnx_sess = ort.InferenceSession(primary_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, input_fs, fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            if not self.only_mos:
                mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
                mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
                predicted_mos_sig_seg_raw.append(mos_sig_raw)
                predicted_mos_bak_seg_raw.append(mos_bak_raw)
                predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
                predicted_mos_sig_seg.append(mos_sig)
                predicted_mos_bak_seg.append(mos_bak)
                predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'fname': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        if not self.only_mos:
            clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
            clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
            clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
            clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
            clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
            clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict

def evaluate_quality(data_path, output_path, personalized_MOS=False, only_mos=False):
    
    models = glob.glob(os.path.join(data_path, "*"))
    audio_clips_list = []
    p808_model_path = os.path.join(WORK_DIR, 'DNSMOS', 'model_v8.onnx')

    # personalized means that interfering speaker is pernalized
    # you can use this with just -p option
    if personalized_MOS:
        primary_model_path = os.path.join(WORK_DIR, 'pDNSMOS', 'sig_bak_ovr.onnx')
    else:
        primary_model_path = os.path.join(WORK_DIR, 'DNSMOS', 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path, only_mos)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    print(len(clips))
    is_personalized_eval = personalized_MOS
    desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(data_path, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    # 이 부분 잘 모름
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    quality_df = pd.DataFrame(rows)
    quality_df.set_index('fname', inplace=True)
    quality_df.sort_index(inplace=True)
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_df.set_index('fname', inplace=True)
        '''
        같은 데이터를 여러번 추론하는 경우 변경된 부분만 업데이트하는 조건 문
        인덱스가 같다는 것은 같은 데이터를 추론했다는 뜻
        예시
        1) csv 파일에 noise detect 결과와 DNSMOS 결과 둘다 있는 경우 noise detect 결과는 그대로 두고 DNSMOS만 갱신
        2) csv 파일에 noise detect 결과만 있는 경우 DNSMOS 결과를 오른쪽 컬럼에 이어 붙임
        '''
        if len(existing_df.index) == len(quality_df.index) and all(existing_df.index == quality_df.index):
            # P808_MOS 컬럼이 없다는 것은 noise_detect 결과만 있다는 뜻
            # 따라서 단순히 concat하면 됨
            if 'P808_MOS' not in existing_df.columns:
                quality_df = pd.concat([existing_df, quality_df], axis=1)
            # state는 noise detector의 결과 컬럼임
            # 따라서 DNSMOS df 가장 왼쪽에 state 컬럼을 만들고 붙여넣기
            elif 'state' in existing_df.columns:
                quality_df.insert(0, 'state', existing_df['state'])
        # 위 조건문에 걸리지 않았다면 다른 데이터를 같은 이름의 csv 파일로 추론했다는 뜻이므로 기존 csv 결과물을 새 csv 결과물로 완전 대체
        else:
            pass

    quality_df.to_csv(output_path)
    os.chmod(output_path, 0o0777)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.', 
                        help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('-p', "--personalized_MOS", action='store_true', 
                        help='Flag to indicate if personalized MOS score is needed or regular')
    
    args = parser.parse_args()

    evaluate_quality(args.testset_dir, args.csv_path, args.personalized_MOS)
