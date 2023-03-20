#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk
import numpy as np
import torch
import torch.nn as nn
import yaml
from tokenizers import Tokenizer
from transformers import BertTokenizer, AutoConfig, BertConfig, BertLMHeadModel, AutoTokenizer, PreTrainedTokenizerFast, \
    BartTokenizer, BartConfig
import os
from gensim.models.word2vec import Word2Vec
# from data_handling.WordTokenizer import WordTokenizer
from models.audio_encoder_config import AudioEncoderConfig
from models.audio_encoder import AudioEncoderModel
from models.configuration_audio_encoder_decoder import AudioEncoderDecoderConfig
from models.modeling_audio_encoder_decoder import AudioEncoderDecoderModel


class BertCaptionModel(nn.Module):

    def __init__(self, config):
        super(BertCaptionModel, self).__init__()
        self.config = config
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],
                                            audio_args=config["audio_args"])
        self.encoder = AudioEncoderModel(encoder_config)

        decoder_name = config["text_decoder_args"]["name"]
        decoder_pretrained = config["text_decoder_args"]["pretrained"]

        self.tokenizer = BertTokenizer.from_pretrained(decoder_name)
        if decoder_pretrained:
            decoder_config = AutoConfig.from_pretrained(config["text_decoder_args"]["name"],
                                                        add_cross_attention=True,
                                                        is_decoder=True)
        else:
            config["text_decoder_args"]["vocab_size"] = self.tokenizer.vocab_size
            decoder_config = BertConfig(**config["text_decoder_args"]["bert_args"])

        self.model_config = AudioEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config,
                                                                                   decoder_config)
        self.model_config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model = AudioEncoderDecoderModel(config=self.model_config,
                                              is_pretrained=False)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward_encoder(self, audios):
        # samples: audio features
        outputs = self.model.encoder(audios)
        return outputs.last_hidden_state

    def forward_decoder(self, text, audio_embeds):
        # samples: raw texts
        text = self.tokenizer(text,
                              padding="longest",
                              truncation=True,
                              max_length=30,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)
        decoder_targets = input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100)

        decoder_targets[:, 0] = -100

        # audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(self.device)

        decoder_output = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=None,
            labels=decoder_targets,
            return_dict=True
        )

        return decoder_output, decoder_targets

    def forward(self, audios, text):
        text = self.tokenizer(text,
                              padding="longest",
                              truncation=True,
                              max_length=30,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100)

        decoder_targets[:, 0] = -100

        decoder_output = self.model(
            audio_feats=audios,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            labels=decoder_targets,
            return_dict=True
        )
        return decoder_output.loss

    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=3,
                 max_length=30,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):
        # samples: audios

        if use_nucleus_sampling:
            outputs = self.model.generate(
                inputs=samples,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                decoder_start_token_id=self.model_config.decoder_start_token_id
                    )
        else:
            outputs = self.model.generate(
                inputs=samples,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                decoder_start_token_id=self.model_config.decoder_start_token_id
            )

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


if __name__ == '__main__':
    os.chdir("../")
    with open("settings/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = BertCaptionModel(config)
    audio_feats = torch.randn(16, 1, 64, 100)
    text = ["this is a sample" for i in range(16)]
    output = model.generate(audio_feats)
    print(output)
