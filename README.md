# WavCaps
A large-scale weakly-labelled audio captioning dataset for audio-language multimodal research.

### Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Codes and Models](#codes)
5. [Citation](#citation)

## Introduction

Audio-language (AL) multi-modal learning tasks have made significant strides in recent years, but the limited size of existing audio-language datasets remains a challenge for researchers due to the expensive and time-consuming collection process. 
To address this data scarcity problem, we present WavCaps, a large-scale weakly-labelled audio captioning dataset containing approximately 400k audio clips and paired captions.
We harvested audio clips and their raw-descriptions from multiple web sources and a sound event detection dataset, and employed ChatGPT, a large language model (LLM), to process the crawled raw-descriptions, including filtering and transforming them into caption-like sentences.

We conduct a comprehensive analysis of the characteristics of our proposed WavCaps dataset and evaluate it on multiple downstream audio-language multimodal learning tasks. By pretraining the models on WavCaps, we achieve new state-of-the-art (SOTA) on  tasks including audio-language retrieval, automated audio captioning, and zero-shot audio classification, outperforming previous SOTA by a significant margin. We hope that our proposed WavCaps dataset could facilitate the research in audio-language multimodal learning and sever as an example of leveraging ChatGPT to enrich academic research.

This repository contains the dataset, and source codes for downstream tasks.

## Dataset

WavCaps are sourced from three websites and a sound event detection dataset:
* [FreeSound](https://freesound.org/)
* [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/)
* [SoundBible](https://soundbible.com/)
* [AudioSet Strongly-labelled Subset](https://research.google.com/audioset/download_strong.html)

[ChatGPT](https://openai.com/blog/chatgpt) is leveraged to process and transform raw-descriptions into captions.

We release WavCaps dataset under json or csv formats, where the key `description` contains raw-description and `caption` contains our processed caption.
These files are under `data` directory.

### License
Only academic uses are allowed for WavCaps dataset. By downloading audio clips through the links provided in the json files, you agree that you will use the audios for research purposes only.
For credits for audio clips from FreeSound, please refer to its own page.

For detailed license information, please refer to:
[FreeSound](https://freesound.org/help/faq/#licenses), [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/licensing), [SoundBible](https://soundbible.com/about.php)


## Codes
We provide codes and pre-trained models for audio-language retrieval, automated audio captioning, and zero-shot audio classification.

* [Retrieval]()
* [Captioning]()
* [Zero-shot Audio Classification]()
* [Text-to-Sound Generation](https://github.com/haoheliu/AudioLDM)

## Citation

Please cite our paper as below if you use the WavCaps dataset.





