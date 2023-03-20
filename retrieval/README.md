## Codes for Audio-Language Retrieval

### Download pretrained audio encoders
Please download pretrained audio encoders. [Google Drive](https://drive.google.com/drive/folders/1ZaYERuMMLLgu4oHTl47FcippLFboaGq5?usp=share_link)
Put them under `pretrained_models/audio_encoders`

### Configure training files
* You can configure training settings in yaml files under `settings` directory.

* For our dataloader, we use json files, and the `audio` key refers to the path of the audio clip in your computer or server.

* Run `pretrain.py` for pretraining, and `train.py` for finetuning or training from scratch. 

### Pretrained Models
We provide pretrained audio-language retrieval models for reproducing results from.

Pretrained models can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1pFr8IRY3E1FAtc2zjYmeuSVY3M5a-Kdj?usp=share_link)
