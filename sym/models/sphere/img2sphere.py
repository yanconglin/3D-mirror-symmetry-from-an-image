import os
import math
import numpy as np
import warnings

import numpy.linalg as LA
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
from glob import glob
from .sphere_utils import w2S, gold_spiral_sampling_patch, cos_cdis



def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, "cpp_img2sphere")
    tar_dir = os.path.join(src_dir, "build", ext_name)
    os.makedirs(tar_dir, exist_ok=True)
    srcs = glob(f"{src_dir}/*.cu") + glob(f"{src_dir}/*.cpp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.utils.cpp_extension import load
        ext = load(
            name=ext_name,
            sources=srcs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[],
            build_directory=tar_dir,
        )
    return ext

# defer calling load_cpp_ext to make CUDA_VISIBLE_DEVICES happy
img2sphere = None


class IMG2SPHERE_Function(Function):
    @staticmethod
    def forward(ctx,
                atten, 
                gamma,
                planes,
                planes_d, 
                side_flag
                ):
        B, HW, _ = atten.shape
        ctx.HW = HW
        ctx.D = len(gamma)
        _, S, _, _= planes.shape
        ctx.S = S
        ctx.side_flag = side_flag
        ctx.save_for_backward(gamma, planes, planes_d)
        output = img2sphere.img2sphere_forward(
            atten,
            gamma,
            planes,
            planes_d, 
            side_flag
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        gamma = ctx.saved_tensors[0]
        planes = ctx.saved_tensors[1]
        planes_d = ctx.saved_tensors[2]

        grad_atten = img2sphere.img2sphere_backward(
            grad_output,
            gamma,
            planes,
            planes_d,
            ctx.side_flag
        )

        return ( 
            grad_atten,  # atten
            None, # gamma
            None,  # planes
            None,  # planes_d
            None  # side_flag
        )


class IMG2SPHERE(nn.Module):
    def __init__(self, D, depth_min, depth_max, side_flag=0.0):
        super(IMG2SPHERE, self).__init__()

        global img2sphere
        print('#################### img2sphere compiling ############################')
        img2sphere = load_cpp_ext("img2sphere")
        print('#################### img2sphere compiling done! ############################')
        self.D=D
        self.depth_min=depth_min
        self.depth_max=depth_max
        self.side_flag=0.0
        self.register_buffer('gamma', torch.linspace(depth_min, depth_max, D, dtype=torch.float32))
        self.__repr__()

    def __repr__(self):
        return self.__class__.__name__


    def forward(self, atten, planes, planes_d):
        # input: attn [B, HW, HW]
        # print('input', atten.shape, planes.shape, planes_d.shape)
        output = IMG2SPHERE_Function.apply(
            atten.contiguous(),
            self.gamma.contiguous(),
            planes.contiguous(),
            planes_d.contiguous(),
            self.side_flag
        )
        return output


