#pragma once
#include <torch/extension.h>


at::Tensor
img2sphere_forward(
            const at::Tensor &input,
            const at::Tensor &gamma,
            const at::Tensor &planes,
            const at::Tensor &planes_p,
            const float side_flag
            );

at::Tensor
// std::vector<at::Tensor>
img2sphere_backward(
                const at::Tensor &grad_output,
                const at::Tensor &gamma,
                const at::Tensor &planes,
                const at::Tensor &planes_p,
                const float side_flag
                );
