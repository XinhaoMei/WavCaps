#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


from torch import optim
import numpy as np


def get_optimizer(params, lr, betas, eps, momentum, optimizer_name):
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            params, lr=lr, betas=betas, eps=eps
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            params, lr=lr, momentum=momentum
        )
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            params, lr=lr, betas=betas, eps=eps
        )
    else:
        raise ValueError("optimizer name is not correct")
    return optimizer


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def step_lr(optimizer, base_lr, warmup_length, adjust_steps, gamma):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if (step - warmup_length) > 0 and (step - warmup_length) % adjust_steps == 0:
                lr = optimizer.param_groups[0]["lr"] * gamma
            else:
                lr = optimizer.param_groups[0]["lr"]
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster