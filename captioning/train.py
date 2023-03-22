#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import argparse
import torch
import time
import wandb
from pprint import PrettyPrinter
import torch
import platform
import argparse
import ruamel.yaml as yaml
from loguru import logger
from warmup_scheduler import GradualWarmupScheduler
from data_handling.datamodule import AudioCaptionDataModule
from models.bart_captioning import BartCaptionModel
from models.bert_captioning import BertCaptionModel
from pretrain import validate, train
from tools.optim_utils import get_optimizer, cosine_lr, step_lr
from tools.utils import setup_seed, set_logger


def main():
    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='htsat_test', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', default='settings/settings.yaml', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-l', '--lr', default=1e-04, type=float,
                        help='Learning rate.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed.')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["exp_name"] = args.exp_name
    config["seed"] = args.seed
    config["optim_args"]["lr"] = args.lr

    setup_seed(config["seed"])

    exp_name = config["exp_name"]

    folder_name = '{}_lr_{}_batch_{}_seed_{}'.format(exp_name,
                                                     config["optim_args"]["lr"],
                                                     config["data_args"]["batch_size"],
                                                     config["seed"])

    model_output_dir, log_output_dir = set_logger(folder_name)

    main_logger = logger.bind(indent=1)

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    # data loading
    datamodule = AudioCaptionDataModule(config, config["data_args"]["dataset"])
    train_loader = datamodule.train_dataloader(is_distributed=False)
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    if "bart" in config["text_decoder_args"]["name"]:
        model = BartCaptionModel(config)
    elif "bert" in config["text_decoder_args"]["name"]:
        model = BertCaptionModel(config)
    model = model.to(device)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    wandb.init(
        project="audio-captioning",
        name=folder_name,
        config=config
    )

    wandb.watch(model)

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')

    if config["pretrain"]:
        pretrain_checkpoint = torch.load(config["pretrain_path"])
        model.load_state_dict(pretrain_checkpoint["model"])
        main_logger.info(f"Loaded weights from {config['pretrain_path']}")

    # set up optimizer and loss
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              weight_decay=config["optim_args"]["weight_decay"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    # scheduler = None
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(train_loader),
                          steps=len(train_loader) * config["training"]["epochs"])
    # scheduler = step_lr(optimizer,
    #                     base_lr=config["optim_args"]["lr"],
    #                     warmup_length=config["optim_args"]["warmup_epochs"] * len(train_loader),
    #                     adjust_steps=config["optim_args"]["step_epochs"] * len(train_loader),
    #                     gamma=config["optim_args"]["gamma"])
    # scheduler_temp = None
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=config["optim_args"]["lr"], weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    # training loop
    loss_stats = []
    spiders = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        # scheduler_warmup.step()
        train_statics = train(model, train_loader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        # validation loop, validation after each epoch
        main_logger.info("Validating...")
        for i in range(1, 4):
            metrics = validate(val_loader,
                               model,
                               device=device,
                               log_dir=log_output_dir,
                               epoch=epoch,
                               beam_size=i)
            spider = metrics["spider"]["score"]
            if i != 1:
                spiders.append(spider)

                if spider >= max(spiders):
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": i,
                        "epoch": epoch,
                        "config": config,
                    }, str(model_output_dir) + '/best_model.pt'.format(epoch))

    # Training done, evaluate on evaluation set
    main_logger.info('Training done. Start evaluating.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pt')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint occurred in {best_epoch} th epoch.')
    for i in range(1, 4):
        spider = validate(test_loader, model,
                          device=device,
                          log_dir=log_output_dir,
                          epoch=0,
                          beam_size=i,
                          )['spider']['score']
        wandb.log({f"test/spider(beam: {i})": spider})
    main_logger.info('Evaluation done.')
    wandb.finish()


if __name__ == '__main__':
    main()
