#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import time
from pprint import PrettyPrinter
import wandb
import numpy as np
import torch
import argparse
import ruamel.yaml as yaml
from tqdm import tqdm
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from models.ase_model import ASE
import torch.distributed as dist

from pretrain import validate
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_seed,
    AverageMeter, t2a, a2t, set_logger, log_results,
)


def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()

    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    for batch_id, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):

        optimizer.zero_grad()

        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        loss = model(audio, text, idx)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="settings/train.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="name of this experiment.")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                        help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,
                        help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int,
                        help="Max length.")
    parser.add_argument("-d", "--dataset", default="AudioCaps", type=str,
                        help="Dataset.")

    args = parser.parse_args()

    exp_name = args.exp_name

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["audio_encoder_args"]["type"] = args.model_type
    config["audio_encoder_args"]["model"] = args.model
    config["audio_args"]["max_length"] = args.max_length

    # setup distribution mode
    init_distributed_mode(config["dist_args"])
    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"] + get_rank()
    setup_seed(seed)

    config["optim_args"]["lr"] = args.lr
    config["data_args"]["dataset"] = args.dataset
    exp_name = exp_name + f"_{args.dataset}_lr_{args.lr}_seed_{seed}"

    wandb.init(
        project="AT-retrieval",
        name=exp_name,
        config=config
    )

    # load evaluation datamodule

    datamodule = AudioCaptionDataModule(config, args.dataset)

    dataloader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # setup model
    model = ASE(config)
    model = model.to(device)
    wandb.watch(model)

    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])
    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    if config["resume"]:
        cp = torch.load(config["checkpoint"], map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        model.load_state_dict(state_dict)
    elif config["pretrain"]:
        cp = torch.load(config["pretrain_path"], map_location=device)
        state_dict = cp["model"]
        model.load_state_dict(state_dict)
        logger.info(f"Loaded pretrain model from {config['pretrain_path']}")

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')

    model_without_ddp = model
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model
        )
        model_without_ddp = model.module

    loss_stats = []
    recall_stats = []
    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        if loss <= min(loss_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch
            }

            torch.save(sav_obj, str(model_output_dir) + "/best_model.pt")

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()

        # validate on AC and Clotho
        metrics = validate(model, val_loader, device)
        log_results(metrics, config['data_args']['dataset'], main_logger, test=False)
        recall_stats.append(metrics["t2a"][0] + metrics["a2t"][0])
        if recall_stats[-1] >= max(recall_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch
            }
            torch.save(sav_obj, str(model_output_dir) + "/recall_best_model.pt")

    main_logger.info('Evaluation start...')
    test_loader = datamodule.test_dataloader()

    model.load_state_dict(torch.load(str(model_output_dir) + "/best_model.pt")["model"])
    main_logger.info(f"Evaluation model with smallest loss...epoch:{torch.load(str(model_output_dir) + '/best_model.pt')['epoch']}")
    metrics = validate(model, test_loader, device)
    log_results(metrics, config['data_args']['dataset'], main_logger, test=True)

    model.load_state_dict(torch.load(str(model_output_dir) + "/recall_best_model.pt")["model"])
    main_logger.info(
        f"Evaluation model with highest recall...epoch:{torch.load(str(model_output_dir) + '/recall_best_model.pt')['epoch']}")
    metrics = validate(model, test_loader, device)
    log_results(metrics, config['data_args']['dataset'], main_logger, test=True)

    main_logger.info("Done.")
    wandb.finish()


if __name__ == '__main__':
    main()
