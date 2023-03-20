#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import random

import librosa
import numpy
import torch
import re
from torch.utils.data import Dataset
from data_handling.text_transform import text_preprocess


class AudioCaptionDataset(Dataset):

    def __init__(self,
                 audio_config: dict,
                 dataset: str = "AudioCaps",
                 split: str = "train",
                 ):
        super(AudioCaptionDataset, self).__init__()
        self.dataset = dataset
        self.split = split
        self.sr = audio_config["sr"]
        json_path = f"data/{dataset}/json_files/{split}.json"

        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0

        with open(json_path, 'r') as f:
            json_obj = json.load(f)["data"]

        if self.dataset == "AudioCaps" and split == "train":
            self.num_captions_per_audio = 1
            self.captions = [item["caption"] for item in json_obj]
            self.wav_paths = [item["audio"] for item in json_obj]
        elif split == "train":
            self.num_captions_per_audio = 5
            self.captions = [item["caption_{}".format(i)] for item in json_obj for i in range(1, 6)]
            self.wav_paths = [item["audio"] for item in json_obj for _ in range(1, 6)]
        else:
            self.num_captions_per_audio = 5
            self.captions = []
            for item in json_obj:
                captions = [item["caption_{}".format(i)] for i in range(1, 6)]
                self.captions.append(captions)
                self.wav_paths = [item["audio"] for item in json_obj]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        if self.split in ["val", "test"]:
            audio_idx = index
        else:
            audio_idx = index // self.num_captions_per_audio

        audio_name = self.wav_paths[index].split("/")[-1]
        wav_path = self.wav_paths[index]

        # load waveform
        waveform, sr = librosa.load(wav_path, sr=self.sr, mono=True)

        if self.max_length != 0:
            # if audio length is longer than max_length, we random crop it
            if waveform.shape[-1] > self.max_length:
                max_start = waveform.shape[-1] - self.max_length
                start = random.randint(0, max_start)
                waveform = waveform[start: start + self.max_length]

        if isinstance(self.captions[index], list):
            caption = [text_preprocess(cap) for cap in self.captions[index]]
        else:
            caption = text_preprocess(self.captions[index])

        return torch.from_numpy(waveform), caption, audio_name, audio_idx
