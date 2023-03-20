#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
from models.cnns import ResNet38, Cnn14
from models.htsat import HTSAT_Swin_Transformer


class AudioEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config["audio_encoder_args"]["type"] == "cnn":
            if config["audio_encoder_args"]["model"] == 'ResNet38':
                self.audio_enc = ResNet38(config)
            elif config["audio_encoder_args"]["model"] == 'Cnn14':
                self.audio_enc = Cnn14(config)

            if config["audio_encoder_args"]["pretrained"]:
                # loading pretrained CNN weights
                pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                            format(config["audio_encoder_args"]["model"]))['model']
                dict_new = self.audio_enc.state_dict().copy()
                trained_list = [i for i in pretrained_cnn.keys()
                                if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
                for i in range(len(trained_list)):
                    dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
                self.audio_enc.load_state_dict(dict_new)

            self.audio_width = 2048

        elif config["audio_encoder_args"]["type"] == "transformer":
            self.audio_enc = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=527,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=config,
            )
            if config["audio_encoder_args"]["pretrained"]:
                audio_ckpt = torch.load("pretrained_models/audio_encoder/HTSAT.ckpt", map_location="cpu")["state_dict"]
                for key in list(audio_ckpt.keys()):
                    if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                        and 'logmel_extractor' not in key):
                        v = audio_ckpt.pop(key)
                        audio_ckpt[key[10:]] = v
                self.audio_enc.load_state_dict(audio_ckpt, strict=False)
                param_names = [n for n, p in self.audio_enc.named_parameters()]
                # for n in param_names:
                #     print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
            self.audio_width = 768
        else:
            raise NotImplementedError('No such audio encoder network.')

        if config["audio_encoder_args"]["freeze"]:
            for name, param in self.audio_enc.named_parameters():
                param.requires_grad = False

    def forward(self, inputs):
        """

        :param inputs: audio features
        :return: encoded audio embeddings
        """

        audio_encoded = self.audio_enc(inputs)
        return audio_encoded
