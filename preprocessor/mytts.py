import os

import pandas as pd
import librosa
import numpy as np
import re
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    text_dir = config["path"]["script_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "mytts"

    metadata = pd.read_excel(text_dir, dtype='object', header=None)
    wavs = metadata[0].values
    script = metadata[2].values
    filters = '([.,!?])'

    for i in range(len(script)):
        base_name = wavs[i][:-4]
        text = script[i]
        text = _clean_text(text, cleaners)
        text = re.sub(re.compile(filters), '', text)

        wav_path = os.path.join(in_dir, "mytts_wav", "{}.wav".format(base_name))
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)