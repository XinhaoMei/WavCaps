[![arXiv](https://img.shields.io/badge/arXiv-2303.17395-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2303.17395)

:star2: WavCaps can be downloaded through [HuggingFace](https://huggingface.co/datasets/cvssp/WavCaps) now!

# WavCaps
WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research. [(arXiv)](https://arxiv.org/abs/2303.17395)

This repository contains:

- Metadata of WavCaps dataset.
- Source code for related tasks: audio-language retrieval, automated audio captioning, and zero-shot audio classification.

## Table of Contents


- [WavCaps](#wavcaps)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [License](#license)
  - [Code for related tasks](#code-for-related-tasks)
  - [Citation](#citation)



## Introduction

The advancement of audio-language (AL) multimodal learning tasks has been significant in recent years, yet the limited size of existing audio-language datasets poses challenges for researchers due to the costly and time-consuming collection process. 
To address this data scarcity issue, we introduce WavCaps, the first large-scale weakly-labelled audio captioning dataset, comprising approximately 400k audio clips with paired captions. 
We sourced audio clips and their raw descriptions from web sources and a sound event detection dataset.
However, the online-harvested raw descriptions are highly noisy and unsuitable for direct use in tasks such as automated audio captioning.
To overcome this issue, we propose a three-stage processing pipeline for filtering noisy data and generating high-quality captions, where ChatGPT, a large language model, is leveraged to filter and transform raw descriptions automatically. 

We conduct a comprehensive analysis of the characteristics of WavCaps dataset and evaluate it on multiple downstream audio-language multimodal learning tasks. The systems trained on WavCaps outperform previous state-of-the-art (SOTA) models by a significant margin. 
Our aspiration is for the WavCaps dataset we have proposed to facilitate research in audio-language multimodal learning and demonstrate the potential of utilizing ChatGPT to enhance academic research.

## Dataset

WavCaps are sourced from three websites and a sound event detection dataset:
* [FreeSound](https://freesound.org/) (262300)
* [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/) (31201)
* [SoundBible](https://soundbible.com/) (1232)
* [AudioSet Strongly-labelled Subset](https://research.google.com/audioset/download_strong.html) (108317)

[ChatGPT](https://openai.com/blog/chatgpt) is leveraged to process and transform raw-descriptions into captions.

We release WavCaps dataset on [HuggingFace](https://huggingface.co/datasets/cvssp/WavCaps). Each data source is released with a json file and corresponding wavefroms. 

## License
Only academic uses are allowed for WavCaps dataset. By downloading audio clips through the links provided in the json files, you agree that you will use the audios for research purposes only.
For credits for audio clips from FreeSound, please refer to its own page.

For detailed license information, please refer to:
[FreeSound](https://freesound.org/help/faq/#licenses), [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/licensing), [SoundBible](https://soundbible.com/about.php)

The models we provided are created under a UK data copyright exemption for non-commercial research.


## Code for related tasks
We provide codes and pre-trained models for audio-language retrieval, automated audio captioning, and zero-shot audio classification.

* [Retrieval](https://github.com/XinhaoMei/WavCaps/tree/master/retrieval)
* [Captioning](https://github.com/XinhaoMei/WavCaps/tree/master/captioning)
* [Zero-shot Audio Classification](https://github.com/XinhaoMei/WavCaps/blob/master/retrieval/zero_shot_classification.py)
* [Text-to-Sound Generation](https://github.com/haoheliu/AudioLDM)
* [Models](https://drive.google.com/drive/folders/1pFr8IRY3E1FAtc2zjYmeuSVY3M5a-Kdj?usp=share_link)

## Citation

Please cite our paper as below if you use the WavCaps dataset.
```bibtex
@article{mei2023wavcaps,
  title={WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research},
  author={Mei, Xinhao and Meng, Chutong and Liu, Haohe and Kong, Qiuqiang and Ko, Tom and Zhao, Chengqi and Plumbley, Mark D and Zou, Yuexian and Wang, Wenwu},
  journal={arXiv preprint arXiv:2303.17395},
  year={2023}
}
```




