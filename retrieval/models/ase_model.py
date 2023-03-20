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

        # create momentum models
        # self.momentum = 0.995
        # self.audio_encoder_m = copy.deepcopy(self.audio_encoder)
        # self.audio_proj_m = copy.deepcopy(self.audio_proj)
        # self.text_encoder_m = copy.deepcopy(self.text_encoder)
        # self.text_proj_m = copy.deepcopy(self.text_proj)
        #
        # remove_grad(self.audio_encoder_m)
        # remove_grad(self.audio_proj_m)
        # remove_grad(self.text_encoder_m)
        # remove_grad(self.text_proj_m)
        # self.model_pairs = [
        #     [self.audio_encoder, self.audio_encoder_m],
        #     [self.audio_proj, self.audio_proj_m],
        #     [self.text_encoder, self.text_encoder_m],
        #     [self.text_proj, self.text_proj_m]
        # ]

        # self.queue_flag = False
        self.temp = nn.Parameter(torch.ones([]) * config["temp"])
        # self.queue_size = config["queue_size"]
        #
        # self.register_buffer("audio_queue", torch.randn(embed_size, self.queue_size))
        # self.register_buffer("text_queue", torch.randn(embed_size, self.queue_size))
        # self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        #
        # self.audio_queue = F.normalize(self.audio_queue, dim=0)
        # self.text_queue = F.normalize(self.text_queue, dim=0)

        self.embed_reg = config["embed_regularization"]

        self.atc_loss = AudioTextContrastiveLoss()
        # self.atc_loss = NTXent()

    def encode_audio(self, audio):
        audio_feats = self.audio_encoder(audio)
        audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        return audio_embeds

    def encode_text(self, text):
        text_feats = self.text_encoder(text)
        text_embeds = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds

    def encode_audio_m(self, audio):
        audio_feats_m = self.audio_encoder_m(audio)
        audio_embeds_m = F.normalize(self.audio_proj_m(audio_feats_m), dim=-1)
        return audio_embeds_m

    def encode_text_m(self, text):
        text_feats_m = self.text_encoder_m(text)
        text_embeds_m = F.normalize(self.text_proj_m(text_feats_m[:, 0, :]), dim=-1)
        return text_embeds_m

    def forward(self, audio, text, idx):

        audio_embeds = self.encode_audio(audio)
        text_embeds = self.encode_text(text)

        idx = idx.view(-1, 1)
        # idx_all = torch.cat([idx.t(), self.idx_queue.detach().clone()], dim=1)
        # pos_idx = torch.eq(idx, idx_all).float()
        pos_idx = torch.eq(idx, idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # with torch.no_grad():
        #     self._momentum_update()
        #     audio_embeds_m = self.encode_audio_m(audio)
        #     text_embeds_m = self.encode_text_m(text)
        #     audio_embeds_all = torch.cat(
        #         [audio_embeds_m.t(), self.audio_queue.detach().clone()], dim=1
        #     )
        #
        #     text_embeds_all = torch.cat(
        #         [text_embeds_m.t(), self.text_queue.detach().clone()], dim=1
        #     )
        #
        # sim_a2t = audio_embeds @ text_embeds_all / self.temp
        # sim_t2a = text_embeds @ audio_embeds_all / self.temp
        # loss = self.atc_loss(sim_a2t, sim_t2a, sim_targets)
        sim_a2t = audio_embeds @ text_embeds.t() / self.temp
        sim_t2a = text_embeds @ audio_embeds.t() / self.temp
        loss = self.atc_loss(sim_a2t, sim_t2a, sim_targets)
        if self.embed_reg:
            loss = loss + torch.mean(torch.abs(audio_embeds)) / torch.sqrt(torch.sum(audio_embeds**2)) + \
                   torch.mean(torch.abs(text_embeds)) / torch.sqrt(torch.sum(text_embeds**2))

        # self._dequeue_and_enqueue(audio_embeds_m, text_embeds_m, idx)

        return loss

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, audio_embed, text_embed, idx):
        audio_embeds = _gather_embeddings(audio_embed)
        text_embeds = _gather_embeddings(text_embed)
        idxs = _gather_embeddings(idx)
        batch_size = audio_embeds.shape[0]
        ptr = int(self.queue_ptr)

        assert (
            self.queue_size % batch_size == 0
        ), "queue size should be divisible by batch_size"

        self.audio_queue[:, ptr: ptr + batch_size] = audio_embeds.T
        self.text_queue[:, ptr: ptr + batch_size] = text_embeds.T
        self.idx_queue[:, ptr: ptr + batch_size] = idxs.T
        # if ptr + batch_size >= self.queue_size and not self.queue_flag:
        #     print("Queue filled.")
        #     self.queue_flag = True
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


def _gather_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return embeddings

    embeddings_all_gpus = [
        torch.zeros_like(embeddings) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(embeddings_all_gpus, embeddings)

    return torch.cat(embeddings_all_gpus)
