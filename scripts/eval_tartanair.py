#!/usr/bin/env python3
import sys
sys.path.append('.')

import os
import argparse
import glob

import torch

from raft3d.raft3d import RAFT3D

from utils import normalize_image
from config.config_loader import load_config

from tartanair_dataloader import make_dataloader


RV_WEIGHT = 0.2
DZ_WEIGHT = 100.0


class MetricsAccumulator:
    """ Stores metrics dicts over multiple iterations and computes averages """

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def push(self, metrics):
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1

    def average(self):
        return {key: self.sums[key] / self.counts[key] for key in self.sums}

    def reset(self):
        self.sums = {}
        self.counts = {}


def loss_fn(flow2d_est, flow2d_rev, flow_gt, valid_mask, gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    N = len(flow2d_est)
    loss = 0.0

    for i in range(N):
        w = gamma**(N - i - 1)
        fl_rev = flow2d_rev[i]

        fl_est, dz_est = flow2d_est[i].split([2,1], dim=-1)
        fl_gt, dz_gt = flow_gt.split([2,1], dim=-1)

        loss += w * (valid_mask * (fl_est - fl_gt).abs()).mean()
        loss += w * DZ_WEIGHT * (valid_mask * (dz_est - dz_gt).abs()).mean()
        loss += w * RV_WEIGHT * (valid_mask * (fl_rev - fl_gt).abs()).mean()

    epe_2d = (fl_est - fl_gt).norm(dim=-1)
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)]

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)]

    metrics = {
        'epe2d': epe_2d.mean().item(),
        'epedz': epe_dz.mean().item(),
        '1px': (epe_2d < 1).float().mean().item(),
        '3px': (epe_2d < 3).float().mean().item(),
        '5px': (epe_2d < 5).float().mean().item(),
    }

    return loss, metrics


def fetch_dataloader(args):
    eval_loader = make_dataloader(
        root="/mnt/cps_persistent1_shared/datasets/public/TartanAir/data_v2",
        envs=["ArchVizTinyHouseDay"],
        difficulties=["easy"],
        trajectories=["P000"],
        camera="lcam_back",
        frame_sep=args.frameskip,
        loader_workers=1,
        batch_size=1,
    )
    return eval_loader


@torch.no_grad
def eval(args, config):

    # collect model checkpoints, skipping saved optimizer states
    ckpt_paths = sorted(
        p for p in glob.glob(os.path.join(args.ckpt, '*.pth'))
        if not p.endswith('_optimizer.pth')
    )

    if len(ckpt_paths) == 0:
        raise RuntimeError(f"no checkpoints found in {args.ckpt}")

    eval_loader = fetch_dataloader(args)

    print("dataset size", len(eval_loader))

    iterations = config["iterations"]

    all_metrics = {}

    for ckpt_path in ckpt_paths:
        iteration = int(os.path.splitext(os.path.basename(ckpt_path))[0].rsplit('_', 1)[1])

        print("iteration:", iteration)

        model = torch.nn.DataParallel(RAFT3D(config), device_ids=config["gpus"])
        model.load_state_dict(torch.load(ckpt_path), strict=False)
        model.cuda()
        model.eval()

        ma = MetricsAccumulator()

        dataiter = iter(eval_loader)

        for i in range(len(eval_loader)):

            data_blob = next(dataiter)

            image1, image2, depth1, depth2, flow_gt, intrinsics = [data_blob[x].cuda() for x in ["image1", "image2", "depth1", "depth2", "flowxyz", "intrinsics"]]

            image1 = normalize_image(image1.float())
            image2 = normalize_image(image2.float())

            valid_mask = (depth1 < 255.0).unsqueeze(-1)
            with torch.no_grad():
                flow2d_est, flow2d_rev = model(image1, image2, depth1, depth2, intrinsics, iters=iterations, train_mode=True)
                loss, metrics = loss_fn(flow2d_est, flow2d_rev, flow_gt, valid_mask)

            # print(i, metrics)

            ma.push(metrics)

        metrics_avg = ma.average()
        print(iteration, "average", metrics_avg)

        all_metrics[iteration] = metrics_avg

    write_metrics_csv(all_metrics, 'metrics.csv')


def write_metrics_csv(all_metrics, csv_path):
    import csv

    fieldnames = set()
    for metrics in all_metrics.values():
        fieldnames.update(metrics.keys())
    fieldnames = ['iteration'] + sorted(fieldnames)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in sorted(all_metrics.keys()):
            row = {'iteration': iteration}
            row.update(all_metrics[iteration])
            writer.writerow(row)

    print(f"wrote metrics to {csv_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help='folder containing checkpoints to evaluate')
    parser.add_argument('--config', help='Training configuration file (needed to construct the model)')
    parser.add_argument('--frameskip', type=int, default=1, help='Frame separation for the dataloader')

    args = parser.parse_args()

    config = load_config(args)

    # make sure the config loader does not trigger a checkpoint restore;
    # checkpoints are loaded explicitly in the eval loop instead
    config["checkpoint_load_path"] = None

    print(config)
    eval(args, config)
