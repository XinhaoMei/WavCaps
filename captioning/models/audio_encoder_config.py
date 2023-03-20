#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


from transformers import PretrainedConfig


class AudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an Audio Encoder. It is used to instantiate an
    an Audio Encoder according to the specified arguments, defining the model architecture.
    The audio encoder can be a PANNs model or a HTSAT.
    """
    model_type = "audio_encoder"
    
    def __init__(self,
                 model_arch: str = "cnn",
                 model_name: str = "Cnn10",
                 pretrained: bool = True,
                 freeze: bool = False,
                 spec_augment: bool = True,
                 audio_args: dict = None,
                 **kwargs):
        super(AudioEncoderConfig, self).__init__(**kwargs)
        if model_arch not in ["cnn", "transformer"]:
            raise ValueError(f"Not implemented model type: {model_arch}.")
        if model_name not in ["Cnn10", "Cnn14", "ResNet38", "htsat"]:
            raise ValueError(f"Not implemented model: {model_name}.")

        self.model_arch = model_arch
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.hidden_size = 1024 if model_arch == "cnn" else 768
        self.spec_augment = spec_augment
        self.audio_args = audio_args
        self.num_labels = 0
        