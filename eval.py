#!/usr/bin/env python3
"""Compute vanishing points using corase-to-fine method on the evaluation dataset.
Usage:
    eval.py [options] <yaml-config> <checkpoint>
    eval.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --visualize <outdir>          Output visualization related files
   --suffix <suffix>             File surffix of visualization [default: sym]
   --noimshow                    Do not show result
"""

import math
import os
import os.path as osp
import pprint
import random
import shlex
import subprocess
import sys
import threading

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import numpy.linalg as LA
import skimage.io
import torch
from docopt import docopt
from tqdm import tqdm

import sym
from sym.config import CI, CM, CO, C
from sym.datasets import ShapeNetDataset_test, Pix3dDataset_test, sample_sphere_test, w2S, w2P, cos_cdis
from sym.models import SymmetryNet
from sym.utils import np_eigen_scale_invariant, np_kitti_error
from sym.models.sphere.sphere_utils import gold_spiral_sampling_patch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


def visualize(ws, score, w0):
    w0 /= LA.norm(w0) / 1.02
    ws /= LA.norm(ws, axis=1, keepdims=True) / 1.02
    ax = plt.figure(figsize=(10, 6)).add_subplot(111, projection="3d")

    ax.view_init(27, -22)
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

    # draw a hemisphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="g", alpha=0.3)

    # draw sampled points
    _ = ax.scatter(ws[:, 0], ws[:, 1], ws[:, 2], c=score)
    ax.scatter(w0[0], w0[1], w0[2], c="red", marker="^")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0, 1)
    plt.show()

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    CI.update(C.io)
    CM.update(C.model)
    CO.update(C.optim)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        print("Let's use", torch.cuda.get_device_name(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    checkpoint = torch.load(args["<checkpoint>"])
    model = sym.models.SymmetryNet().to(device)
    model = sym.utils.MyDataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    missing, _ = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    assert len(missing) == 0
    model.eval()

    if CI.dataset == "ShapeNet":
        Dataset = ShapeNetDataset_test
    else:
        Dataset =  Pix3dDataset_test

    split = "test_all"
    loader = torch.utils.data.DataLoader(
        Dataset(C.io.datadir, split=split),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    fpath = args["--output"]
    print("save to", fpath)

    w_pd = []
    w_gt = []
    err_normal = []
    with torch.no_grad():
        for batch_idx, input_dict in enumerate(tqdm(loader)):
            result = model(input_dict, mode='test')
            best_w = result["best_w"].cpu().numpy()
            w0 = input_dict["w0"].squeeze(0).cpu().numpy()
            # rescale depth according to the ||w||_2
            Rt = input_dict["Rt"].squeeze(0).cpu().numpy()
            best_w /= abs(Rt[2][3])
            w0 /= abs(Rt[2][3])

            if args["--visualize"]:
                # visualize(ws_pd, scores_pd, w0)
                fname = input_dict["fname"].cpu().numpy()[0].tobytes().decode("ascii")
                fname = fname.rstrip("\x00")[::-1].replace("/", "_", 1)[::-1]
                # print('fname replace', fname)
                fname_pd = f"{args['--visualize']}/{fname}_{args['--suffix']}.npz"
                os.makedirs(osp.dirname(fname_pd), exist_ok=True)
                np.savez(fname_pd, w_gt=w0, w_pd=best_w)

            err_normal += [np.arccos(min(1, abs(best_w @ w0 / LA.norm(w0) / LA.norm(best_w))))]
            w_pd += [best_w]
            w_gt += [w0]
    
    err_normal = np.sort(np.array(err_normal)) / np.pi * 180
    y = (1 + np.arange(len(err_normal))) / len(err_normal)
    print("  | ".join([f"{AA(err_normal, y, th):.3f}" for th in [1, 3, 5]]))
    np.savez(
        fpath,
        err_normal=err_normal,
        w_pd=np.array(w_pd),
        w_gt=np.array(w_gt),
    )





if __name__ == "__main__":
    main()
