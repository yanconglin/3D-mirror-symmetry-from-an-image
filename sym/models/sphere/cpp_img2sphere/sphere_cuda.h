#pragma once
#include <torch/extension.h>


at::Tensor
sphere_cuda_forward(
                const at::Tensor &input,
                const at::Tensor &gamma,
                const at::Tensor &planes,
                const at::Tensor &planes_d,
                const float side_flag
                );

at::Tensor
// std::vector<at::Tensor>
sphere_cuda_backward(
                const at::Tensor &grad_output, 
                const at::Tensor &gamma,
                const at::Tensor &planes,
                const at::Tensor &planes_d,
                const float side_flag
                );

