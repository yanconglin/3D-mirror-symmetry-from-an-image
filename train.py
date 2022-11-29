#!/usr/bin/env python3
"""Training and Evaluate the Neural Network
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
    yaml-config                      Path to the yaml hyper-parameter file

Options:
   -h --help                         Show this screen.
   -d --devices <devices>            Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>      Folder name [default: default-identifier]
   --from <checkpoint>               Path to a checkpoint
   --ignore-optim                    Ignore optim when restoring from a checkpoint
"""

import datetime
import glob
import os
import os.path as osp
import platform
import pprint
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading

import numpy as np
import torch
import yaml
from docopt import docopt

import sym
from sym.config import CI, CM, CO, C, load_config
from sym.datasets import ShapeNetDataset, Pix3dDataset


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    # name += "-%s" % git_hash()
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(CI.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.to_yaml(osp.join(outdir, "config.yaml"))
    # os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    os.system(f"find -name '*.py' -print0 | tar -cJf {outdir}/src.tar.xz --null -T -")
    return outdir


def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/shapenet.yaml"
    C.update(C.from_yaml(filename=config_file))
    if args["--from"]:
        C.io.resume_from = args["--from"]
    CI.update(C.io)
    CM.update(C.model)
    CO.update(C.optim)
    pprint.pprint(C, indent=4)
    resume_from = CI.resume_from

    # WARNING: still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    num_gpus = args["--devices"].count(",") + 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        print("Let's use", torch.cuda.get_device_name(), "GPU(s)!")

    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset
    batch_size = CM.batch_size * num_gpus
    num_workers = CI.num_workers * num_gpus
    datadir = CI.datadir
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if CI.dataset == "ShapeNet":
        Dataset = ShapeNetDataset
    elif CI.dataset == "Pix3d":
        Dataset = Pix3dDataset
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="train"), shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="valid"), shuffle=False, **kwargs
    )
    epoch_size = len(train_loader)
    print('epoch_size: train/valid',  len(train_loader), len(val_loader))

    # 2. model
    model = sym.models.SymmetryNet().to(device)
    model = sym.utils.MyDataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    # print('model', model)
    print("# of params:", count_parameters(model))
    print("# of learnable params:", count_learnable_parameters(model))
    # 3. optimizer
    if CO.name == "Adam":
        optim = torch.optim.Adam(model.parameters(), **CO.params)
    elif CO.name == "SGD":
        optim = torch.optim.SGD(model.parameters(), **CO.params)
    else:
        raise NotImplementedError

    if resume_from:
        print('resume_from', resume_from)
        checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
    outdir = resume_from or get_outdir(args["--identifier"])
    print("outdir:", outdir)

    try:
        trainer = sym.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=batch_size,
            out=outdir,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.epoch = checkpoint["epoch"]
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            print('trainer.epoch, trainer.iteration, trainer.best_mean_loss ', trainer.epoch, trainer.iteration,
                  trainer.best_mean_loss)
            del checkpoint
        trainer.train()
        print('finish trainig at: ', str(datetime.datetime.now()))
    except BaseException:
        # if len(glob.glob(f"{outdir}/viz/*")) <= 1:
        #     shutil.rmtree(outdir)
        raise

if __name__ == "__main__":
    main()
