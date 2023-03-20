#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import torch
import torch.nn as nn
from sentence_transformers import util
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeds, text_embeds, labels):
        """

        :param audio_embeds:
        :param text_embeds:
        :param labels:
        :return:
        """

        n = audio_embeds.size(0)  # batch size

        # dist = []
        sim_a2t = util.cos_sim(audio_embeds, text_embeds)  # (batch_size, x batch_size)
        sim_ap = torch.diag(sim_a2t).view(n, 1)
        d1 = sim_ap.expand_as(sim_a2t)
        d2 = sim_ap.t().expand_as(sim_a2t)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim_a2t - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim_a2t - d2)

        # clear diagonals
        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(cost_a.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        cost_s = cost_s.max(1)[0]
        cost_a = cost_a.max(0)[0]

        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


class BiDirectionalRankingLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(BiDirectionalRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeds, text_embeds, labels):
        """

        :param audio_embeds: (batch_size, embed_dim)
        :param text_embeds: (batch_size, embed_dim)
        :param labels: (batch_size, )
        :return:
        """

        n = audio_embeds.size(0)  # batch size

        # dist = []
        sim_a2t = util.cos_sim(audio_embeds, text_embeds)  # (batch_size, x batch_size)
        sim_ap = torch.diag(sim_a2t).view(n, 1)
        d1 = sim_ap.expand_as(sim_a2t)
        d2 = sim_ap.t().expand_as(sim_a2t)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim_a2t - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim_a2t - d2)

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(cost_a.device)

        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


class NTXent(nn.Module):

    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature

    def forward(self, audio_embeds, text_embeds, labels):

        n = audio_embeds.shape[0]

        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(a2t.device)
        mask_diag = mask.diag()
        mask_diag = torch.diag_embed(mask_diag)
        mask = mask ^ mask_diag

        a2t_loss = - self.loss(a2t).masked_fill(mask, 0).diag().mean()
        t2a_loss = - self.loss(t2a).masked_fill(mask, 0).diag().mean()

        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss


class WeightTriplet(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2):
        super(WeightTriplet, self).__init__()
        self.margin = margin

    def polyloss(self, sim_mat, label):
        epsilon = 1e-5
        size = sim_mat.size(0)
        hh = sim_mat.t()

        loss = list()
        for i in range(size):
            pos_pair_ = sim_mat[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)
        for i in range(size):
            pos_pair_ = hh[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = hh[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)

            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / size
        return loss

    def forward(self, audio_embeds, text_embeds, labels):
        # compute image-sentence score matrix
        scores = util.cos_sim(audio_embeds, text_embeds)
        loss = self.polyloss(scores, labels)
        return loss

