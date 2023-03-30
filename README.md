# WavCaps
WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research.

### Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Codes and Models](#codes)
5. [Citation](#citation)

## Introduction

The advancement of audio-language (AL) multimodal learning tasks has been significant in recent years. 
However, researchers face challenges due to the costly and time-consuming collection process of existing audio-language datasets, which are limited in size. 
To address this data scarcity issue, we introduce WavCaps, the first large-scale weakly-labelled audio captioning dataset, comprising approximately 400k audio clips with paired captions.
We sourced audio clips and their raw descriptions from web sources and a sound event detection dataset. 
However, the online-harvested raw descriptions are highly noisy and unsuitable for direct use in tasks such as automated audio captioning.
To overcome this issue, we propose a three-stage processing pipeline for filtering noisy data and generating high-quality captions, where ChatGPT, a large language model, is leveraged to filter and transform raw descriptions automatically. 

We conduct a comprehensive analysis of the characteristics of WavCaps dataset and evaluate it on multiple downstream audio-language multimodal learning tasks. The systems trained on WavCaps outperform previous state-of-the-art (SOTA) models by a significant margin. 
Our aspiration is for the WavCaps dataset we have proposed to facilitate research in audio-language multimodal learning and demonstrate the potential of utilizing ChatGPT to enhance academic research.

This repository contains the dataset, and source codes for downstream tasks.

## Dataset

WavCaps are sourced from three websites and a sound event detection dataset:
* [FreeSound](https://freesound.org/) (262300)
* [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/) (31201)
* [SoundBible](https://soundbible.com/) (1232)
* [AudioSet Strongly-labelled Subset](https://research.google.com/audioset/download_strong.html) (108317)

[ChatGPT](https://openai.com/blog/chatgpt) is leveraged to process and transform raw-descriptions into captions.

We release WavCaps dataset under json or csv formats, where the key `description` contains raw-description and `caption` contains our processed caption.
These files are under [data](https://github.com/XinhaoMei/WavCaps/tree/master/data) directory (excluding FreeSound) or can be downloaded through [here](https://drive.google.com/drive/folders/1h9P4_qiNVZR-PIZrL5Ow0v62S8C4ygyo?usp=share_link).

For audio clips from AudioSet and BBC Sound Effects, we do not provide download links. Please refer to their websites for downloading.

For audio clips from FreeSound and SoundBible, we provide metadata including original file names, raw descriptions, and links.
In addition, we also provide waveforms for audio clips from FreeSound whose duration is less than 2 seconds.

Please first consider downloading waveforms we provide instead of crawling from FreeSound.

### License
Only academic uses are allowed for WavCaps dataset. By downloading audio clips through the links provided in the json files, you agree that you will use the audios for research purposes only.
For credits for audio clips from FreeSound, please refer to its own page.

For detailed license information, please refer to:
[FreeSound](https://freesound.org/help/faq/#licenses), [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/licensing), [SoundBible](https://soundbible.com/about.php)


## Codes
We provide codes and pre-trained models for audio-language retrieval, automated audio captioning, and zero-shot audio classification.

* [Retrieval](https://github.com/XinhaoMei/WavCaps/tree/master/retrieval)
* [Captioning](https://github.com/XinhaoMei/WavCaps/tree/master/captioning)
* [Zero-shot Audio Classification](https://github.com/XinhaoMei/WavCaps/blob/master/retrieval/zero_shot_classification.py)
* [Text-to-Sound Generation](https://github.com/haoheliu/AudioLDM)

## Citation

Please cite our paper as below if you use the WavCaps dataset.





