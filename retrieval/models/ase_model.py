#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
import torch.nn.functional as F
import copy
from tools.losses import AudioTextContrastiveLoss, NTXent
from tools.utils import remove_grad


class ASE(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)

        # settings for projection layers
        embed_size = config["embed_size"]
        audio_width = self.audio_encoder.audio_width
        text_width = self.text_encoder.text_width

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        self.embed_reg = config["embed_regularization"]

        self.atc_loss = AudioTextContrastiveLoss()

    def encode_audio(self, audio):
        audio_feats = self.audio_encoder(audio)
        audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        return audio_embeds

    def encode_text(self, text):
        text_feats = self.text_encoder(text)
        text_embeds = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds

    def forward(self, audio, text, idx):

        audio_embeds = self.encode_audio(audio)
        text_embeds = self.encode_text(text)

        idx = idx.view(-1, 1)
        pos_idx = torch.eq(idx, idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        sim_a2t = audio_embeds @ text_embeds.t() / self.temp
        sim_t2a = text_embeds @ audio_embeds.t() / self.temp
        loss = self.atc_loss(sim_a2t, sim_t2a, sim_targets)
        if self.embed_reg:
            loss = loss + torch.mean(torch.abs(audio_embeds)) / torch.sqrt(torch.sum(audio_embeds**2)) + \
                   torch.mean(torch.abs(text_embeds)) / torch.sqrt(torch.sum(text_embeds**2))

        return loss
