#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import argparse
import time
from pprint import PrettyPrinter
import torch
import wandb
from loguru import logger
from ruamel import yaml
from tqdm import tqdm
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from eval_metrics import evaluate_metrics
from models.bart_captioning import BartCaptionModel
from models.bert_captioning import BertCaptionModel
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import setup_seed, set_logger, AverageMeter, decode_output


def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()

    epoch_loss = AverageMeter()
    start_time = time.time()

    for batch_id, (audio, text, audio_names, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()

        step = len(dataloader) * (epoch - 1) + batch_id
        if scheduler is not None:
            scheduler(step)

        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio = audio.to(device, non_blocking=True)

        loss = model(audio, text)

        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time

    wandb.log({"loss": epoch_loss.avg,
               "epoch": epoch})
    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


@torch.no_grad()
def validate(data_loader, model, device, log_dir, epoch, beam_size):
    val_logger = logger.bind(indent=1)
    model.eval()
    with torch.no_grad():
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []
        start_time = time.time()

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, caption_dict, audio_names, audio_ids = batch_data
            # move data to GPU
            audios = audios.to(device)

            output = model.generate(samples=audios,
                                    num_beams=beam_size)

            y_hat_all.extend(output)
            ref_captions_dict.extend(caption_dict)
            file_names_all.extend(audio_names)

        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all,
                                                   log_dir, epoch, beam_size=beam_size)
        metrics = evaluate_metrics(captions_pred, captions_gt)

        spider = metrics['spider']['score']
        cider = metrics['cider']['score']

        eval_time = time.time() - start_time

        val_logger.info(f'Cider: {cider:7.4f}')
        val_logger.info(
            f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.1f}')

        if beam_size == 3 and (epoch % 5) == 0:
            for metric, values in metrics.items():
                val_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')

        return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default="settings/pretrain.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="name of this experiment.")
    parser.add_argument('-l', '--lr', default=1e-05, type=float,
                        help='Learning rate.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed.')

    args = parser.parse_args()
    exp_name = args.exp_name

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["exp_name"] = args.exp_name
    config["seed"] = args.seed
    config["optim_args"]["lr"] = args.lr
    seed = config["seed"]
    setup_seed(seed)

    device = torch.device(config["device"])

    exp_name = exp_name + f"lr_{config['optim_args']['lr']}_seed_{seed}"

    wandb.init(
        project="audio-captioning",
        name=exp_name,
        config=config
    )

    model_output_dir, log_output_dir = set_logger(exp_name)
    main_logger = logger.bind(indent=1)

    dataloader = pretrain_dataloader(config,
                                     bucket=True,
                                     bucket_boundaries=(5, 30, 6),
                                     is_distributed=False,
                                     num_tasks=1,
                                     global_rank=0)

    if "bart" in config["text_decoder_args"]["name"]:
        model = BartCaptionModel(config)
    elif "bert" in config["text_decoder_args"]["name"]:
        model = BertCaptionModel(config)
    main_logger.info(f"Decoder model:{config['text_decoder_args']['name']}")
    model = model.to(device)
    wandb.watch(model)

    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              weight_decay=config["optim_args"]["weight_decay"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])

    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')

    # load evaluation datamodule
    ac_datamodule = AudioCaptionDataModule(config, "AudioCaps")
    clotho_datamodule = AudioCaptionDataModule(config, "Clotho")

    ac_val_loader = ac_datamodule.val_dataloader()
    clotho_val_loader = clotho_datamodule.val_dataloader()

    loss_stats = []
    ac_spiders = []
    clotho_spiders = []

    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        # evaluate on AudioCaps
        main_logger.info('Evaluating on AudioCaps...')
        for i in range(1, 4):
            ac_metrics = validate(ac_val_loader,
                                  model,
                                  device=device,
                                  log_dir=log_output_dir,
                                  epoch=epoch,
                                  beam_size=i)
            spider = ac_metrics["spider"]["score"]
            if i != 1:
                ac_spiders.append(spider)
                if spider >= max(ac_spiders):
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": i,
                        "epoch": epoch,
                        "config": config,
                    }, str(model_output_dir) + '/ac_best_model.pt')

            # evaluate on Clotho
        main_logger.info('Evaluating on Clotho...')
        for i in range(1, 4):
            clotho_metrics = validate(clotho_val_loader,
                                      model,
                                      device=device,
                                      log_dir=log_output_dir,
                                      epoch=epoch,
                                      beam_size=i)
            spider = clotho_metrics["spider"]["score"]
            if i != 1:
                clotho_spiders.append(spider)
                if spider >= max(clotho_spiders):
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": i,
                        "epoch": epoch,
                        "config": config,
                    }, str(model_output_dir) + '/clotho_best_model.pt')

    main_logger.info('Training done.')

    ac_test_loader = ac_datamodule.test_dataloader()
    clotho_test_loader = clotho_datamodule.test_dataloader()

    model.load_state_dict(torch.load(str(model_output_dir) + "/ac_best_model.pt")["model"])
    main_logger.info(
        f"Evaluation best AudioCaps model... epoch:{torch.load(str(model_output_dir) + '/ac_best_model.pt')['epoch']}")
    for i in range(1, 4):
        spider = validate(ac_test_loader, model,
                          device=device,
                          log_dir=log_output_dir,
                          epoch=0,
                          beam_size=i,
                          )['spider']['score']
        wandb.log({f"AudioCaps/ac_model/spider(beam: {i})": spider})

    for i in range(1, 4):
        spider = validate(clotho_test_loader, model,
                          device=device,
                          log_dir=log_output_dir,
                          epoch=0,
                          beam_size=i,
                          )['spider']['score']
        wandb.log({f"Clotho/ac_model/spider(beam: {i})": spider})

    model.load_state_dict(torch.load(str(model_output_dir) + "/clotho_best_model.pt")["model"])
    main_logger.info(
        f"Evaluation best Clotho model... epoch:{torch.load(str(model_output_dir) + '/clotho_best_model.pt')['epoch']}")
    for i in range(1, 4):
        spider = validate(ac_test_loader, model,
                          device=device,
                          log_dir=log_output_dir,
                          epoch=0,
                          beam_size=i,
                          )['spider']['score']
        wandb.log({f"AudioCaps/clotho_model/spider(beam: {i})": spider})

    for i in range(1, 4):
        spider = validate(clotho_test_loader, model,
                          device=device,
                          log_dir=log_output_dir,
                          epoch=0,
                          beam_size=i,
                          )['spider']['score']
        wandb.log({f"Clotho/clotho_model/spider(beam: {i})": spider})
    main_logger.info('Evaluation done.')
    wandb.finish()


if __name__ == '__main__':
    main()
