import sys
sys.path.append('.')

import os
import argparse
import cv2
import numpy as np
import warnings
import functools

import torch
import torch.optim as optim

from raft3d.raft3d import RAFT3D

from utils import Logger, show_image, normalize_image, fetch_optimizer, sequence_loss, l1_loss, l2_loss, samplewise_l1_loss, samplewise_l2_loss
from config.config_loader import load_config

from tartanair_dataloader import make_dataloader

VAL_FREQ = 5000
SAVE_FREQ = 5000


def save_model(model, optimizer, config, phase, current_step):
    folder = 'checkpoints'
    if config['checkpoint_save_path'] is not None:
        folder = config['checkpoint_save_path']
    folder = os.path.join(folder, config['name'])
    file = f"{config['name']}_{config['train']['dataset'][phase]}_{current_step:06d}.pth"
    file_optimizer = f"{config['name']}_{config['train']['dataset'][phase]}_{current_step:06d}_optimizer.pth"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    torch.save(model.state_dict(), os.path.join(folder, file))
    torch.save(optimizer.state_dict(), os.path.join(folder, file_optimizer))

def fetch_model(config):
    model = torch.nn.DataParallel(RAFT3D(config), device_ids=config["gpus"])
    if config["checkpoint_load_path"] is not None:
        model.load_state_dict(torch.load(config["checkpoint_load_path"]))

    model.cuda()
    model.train()

    return model


def fetch_optimizer(config, phase, model):
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"][phase], weight_decay=config["train"]["wdecay"][phase], eps=config["adamw_eps"])
    if config["checkpoint_load_path"] is not None:
        optimizer_load_path = f"{config['checkpoint_load_path'][:-4]}_optimizer.pth"
        if os.path.isfile(optimizer_load_path):
            optimizer.load_state_dict(torch.load(optimizer_load_path))

    return optimizer


def train_phase(dataloader, model, optimizer, scheduler, phase, logger, config, validation_func = None):
    num_steps = config["train"]["num_steps"][phase]
    iterations = config["iterations"]

    keep_training = True
    total_steps = 0
    if config["initial_phase"] == phase:
        total_steps = config["initial_step"]

    if config["train"]["loss_fn"][phase] == "l1":
        loss_fn = l1_loss
    elif config["train"]["loss_fn"][phase] == "l2":
        loss_fn = l2_loss
    elif config["train"]["loss_fn"][phase] == "samplewise_l1":
        loss_fn = samplewise_l1_loss
    elif config["train"]["loss_fn"][phase] == "samplewise_l2":
        loss_fn = samplewise_l2_loss
    else:
        raise ValueError(f'Loss function {config["train"]["loss_fn"][phase]} is unknown')

    while keep_training:
        for data_blob in dataloader:
            optimizer.zero_grad()
            # image1, image2, depth1, depth2, flow_gt, valid, intrinsics = [x.cuda() for x in data_blob]

            image1, image2, depth1, depth2, flow_gt, intrinsics = [data_blob[x].cuda() for x in ["image1", "image2", "depth1", "depth2", "flowxyz", "intrinsics"]]
            valid = (depth1 < 255.0).unsqueeze(-1) # TODO: check

            image1 = normalize_image(image1.float())
            image2 = normalize_image(image2.float())

            flow2d_est, flow2d_rev = model(image1, image2, depth1, depth2, intrinsics, iters=iterations, train_mode=True)

            loss, metrics = sequence_loss(flow2d_est, flow2d_rev, flow_gt, valid, loss_fn)

            if torch.isnan(loss):
                print("nan loss during training. Exiting...")
                exit(0)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            metrics.update({"loss": float(loss.float().item())})
            logger.push(metrics)

            total_steps += 1

            if total_steps % SAVE_FREQ == 0 or total_steps >= num_steps:
                save_model(model, optimizer, config, phase, total_steps)

            if validation_func is not None:
                if total_steps % VAL_FREQ == 0 or total_steps >= num_steps:
                    results = validation_func(model.module, iterations)
                    print(results, flush=True)
                    logger.write_dict(results)

            if total_steps >= num_steps:
                keep_training = False
                break

def train(config):
    initial_phase = config["initial_phase"]
    passed_steps = initial_step = config["initial_step"]
    if config["initial_phase"] != 0:
        passed_steps += sum(config["train"]["num_steps"][:config["initial_phase"]])
    num_steps = config["train"]["num_steps"]
    num_phases = len(num_steps)

    learning_rate = config["train"]["lr"]

    model = fetch_model(config)
    logger = Logger(name=config["name"], start_step=passed_steps)

    for phase in range(initial_phase, num_phases):
        optimizer = fetch_optimizer(config, phase, model)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, learning_rate[phase], num_steps[phase], pct_start=0.001, cycle_momentum=False)
        with warnings.catch_warnings():
            # suppress scheduler warning
            warnings.simplefilter("ignore")
            for _ in range(initial_step):
                scheduler.step()

        # TartanAir V2
        dataloader = make_dataloader(
            root="/mnt/cps_persistent1_shared/datasets/public/TartanAir/data_v2",
            envs=["ArchVizTinyHouseDay"],
            difficulties=["easy"],
            trajectories=["P000", "P001", "P002", "P003", "P004", "P005", "P006"],
            camera="lcam_front",
            frame_sep=1,
            loader_workers=1,
            batch_size=2,
        )

        train_phase(dataloader, model, optimizer, scheduler, phase, logger, config)

        initial_step = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Training configuration file')
    parser.add_argument('--ckpt', help='Checkpoint to restore')
    parser.add_argument('--initial_step', type=int, default=0, help='Number of steps the checkpoint has already trained')
    parser.add_argument('--initial_phase', type=int, default=0, help='Number of phases the checkpoint has already trained')
    parser.add_argument('--save', default='checkpoints', help='Folder for saving checkpoints')

    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    config = load_config(args)

    print(config)
    train(config)
