#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
import librosa
import numpy as np
import pandas as pd
import glob
import torch
from ruamel import yaml
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
from re import sub
from data_handling.text_transform import text_preprocess
from models.ase_model import ASE
import torch.nn.functional as F


with open("settings/inference.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda"

model = ASE(config)
model.to(device)

cp_path = 'outputs/pretrain/audio_classification/HTSAT_reg_lr_5e-05_seed_20/models/best_model.pt'
cp = torch.load(cp_path)
model.load_state_dict(cp['model'])
model.eval()
print("Model weights loaded from {}".format(cp_path))

# ESC-50 #########################################

# df = pd.read_csv('data/ESC-50/meta/esc50.csv')
# class_to_idx = {}
# sorted_df = df.sort_values(by=['target'])
# classes = [x.replace('_', ' ') + " can be heard" for x in sorted_df['category'].unique()]
# for i, category in enumerate(classes):
#     class_to_idx[category] = i
# # classes = [c for c in classes]
#
# pre_path = 'data/ESC-50/audio/'
#
# with torch.no_grad():
#     text_embeds = model.encode_text(classes)
#     fold_acc = []
#     for fold in range(1, 6):
#         fold_df = sorted_df[sorted_df['fold'] == fold]
#         y_preds, y_labels = [], []
#         for file_path, target in tqdm(zip(fold_df["filename"], fold_df["target"]), total=len(fold_df)):
#             audio_path = pre_path + file_path
#             one_hot_target = torch.zeros(len(classes)).scatter_(0, torch.tensor(target), 1).reshape(1, -1)
#             audio, _ = librosa.load(audio_path, sr=32000, mono=True)
#             audio = torch.tensor(audio).unsqueeze(0).to(device)
#             if audio.shape[-1] < 32000 * 10:
#                 pad_length = 32000 * 10 - audio.shape[-1]
#                 audio = F.pad(audio, [0, pad_length], "constant", 0.0)
#             audio_emb = model.encode_audio(audio)
#             similarity = audio_emb @ text_embeds.t()
#             y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
#             y_preds.append(y_pred)
#             y_labels.append(one_hot_target.cpu().numpy())
#
#         y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
#         acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
#         print('Fold {} Accuracy {}'.format(fold, acc))
#         fold_acc.append(acc)
#
# print('ESC50 Accuracy {}'.format(np.mean(np.array(fold_acc))))

df = pd.read_csv('data/UrbanSound8K/metadata/UrbanSound8K.csv')
class_to_idx = {}
sorted_df = df.sort_values(by=['classID'])
classes = [x.replace('_', ' ') for x in sorted_df['class'].unique()]
for i, category in enumerate(classes):
    class_to_idx[category] = i
classes = [c for c in classes]

pre_path = 'data/UrbanSound8K/audio/'
with torch.no_grad():
    text_embeds = model.encode_text(classes)
    fold_acc = []
    for fold in range(1, 11):
        fold_df = sorted_df[sorted_df['fold'] == fold]
        y_preds, y_labels = [], []
        for file_path, target in tqdm(zip(fold_df["slice_file_name"], fold_df["classID"]), total=len(fold_df)):
            audio_path = pre_path + file_path
            one_hot_target = torch.zeros(len(classes)).scatter_(0, torch.tensor(target), 1).reshape(1, -1)
            audio, _ = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
            if audio.shape[-1] < 32000 * 10:
                pad_length = 32000 * 10 - audio.shape[-1]
                audio = F.pad(audio, [0, pad_length], "constant", 0.0)
            audio_emb = model.encode_audio(audio)
            similarity = audio_emb @ text_embeds.t()
            y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
            y_preds.append(y_pred)
            y_labels.append(one_hot_target.cpu().numpy())

        y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        print('Fold {} Accuracy {}'.format(fold, acc))
        fold_acc.append(acc)
print('Urbansound8K Accuracy {}'.format(np.mean(np.array(fold_acc))))


# VGGSound  #########################################

# df = pd.read_csv('data/VGGSound/meta/vggsound.csv', names=['filename', 'start_time', 'class', 'split'])
# df_test = df[df['split'] == 'test']
# labels = sorted(df_test['class'].unique())
#
# classes = [text_preprocess(cls) + " can be heard" for cls in labels]
# print(len(classes))
#
# with open('data/VGGSound/meta/vgg_test.json', 'r') as f:
#     vgg_test = json.load(f)["data"]
# pre_path = 'data/VGGSound/audio/'
#
# with torch.no_grad():
#     text_embeds = model.encode_text(classes)
#
#     y_preds, y_labels = [], []
#     for file_path, cls in tqdm(vgg_test, total=len(vgg_test)):
#         audio_path = pre_path + file_path
#         target = labels.index(cls)
#         one_hot_target = torch.zeros(len(classes)).scatter_(0, torch.tensor(target), 1).reshape(1, -1)
#         audio, _ = librosa.load(audio_path, sr=32000, mono=True)
#         audio = torch.tensor(audio).unsqueeze(0).to(device)
#         if audio.shape[-1] < 32000 * 10:
#             pad_length = 32000 * 10 - audio.shape[-1]
#             audio = F.pad(audio, [0, pad_length], "constant", 0.0)
#         elif audio.shape[-1] > 32000 * 10:
#             audio = audio[:, : 32000 * 10]
#         audio_emb = model.encode_audio(audio)
#         similarity = audio_emb @ text_embeds.t()
#         y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
#         y_preds.append(y_pred)
#         y_labels.append(one_hot_target.cpu().numpy())
#
#     y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
#     acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
# print('VGGSound Accuracy {}'.format(np.mean(np.array(acc))))
