import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import argparse

import yaml

from text import _clean_text

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="path to preprocess.yaml")
args = parser.parse_args()

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

in_dir = config["path"]["corpus_path"]
out_dir = config["path"]["raw_path"]
sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
cleaners = config["preprocessing"]["text"]["text_cleaners"]
speaker = "LJSpeech"

with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
    for line in tqdm(f):
        if(int(line[3:5]) < 2):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)
            print(text + "#")