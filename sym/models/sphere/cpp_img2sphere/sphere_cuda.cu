#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "img2sphere_cuda.cuh"


at::Tensor
sphere_cuda_forward(
            const at::Tensor &input,
            const at::Tensor &gamma,
            const at::Tensor &planes,
            const at::Tensor &planes_d,
            const float side_flag
        )
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(gamma.is_contiguous(), "gamma tensor has to be contiguous");
    AT_ASSERTM(planes.is_contiguous(), "planes tensor has to be contiguous");
    AT_ASSERTM(planes_d.is_contiguous(), "planes_d tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(gamma.type().is_cuda(), "gamma must be a CUDA tensor");
    AT_ASSERTM(planes.type().is_cuda(), "planes must be a CUDA tensor");
    AT_ASSERTM(planes_d.type().is_cuda(), "planes_d must be a CUDA tensor");

    const int B = input.size(0);
    const int HW = input.size(1);
    const int HW_ = input.size(2);
    
    const int H = static_cast<int>(sqrt(HW));
    const int W = static_cast<int>(sqrt(HW));

    const int D = gamma.size(0);
    const int S = planes.size(1);

    AT_ASSERTM(planes_d.size(1)==planes.size(1) && planes_d.size(0)==planes.size(0),
        "(P.shape!=S.shape): (%d x %d) vs (%d x %d).", planes.size(0), planes_d.size(0), planes.size(1), planes_d.size(1));

    AT_ASSERTM(HW==HW_, "(input.shape error): (%d vs %d).", HW, HW_);
        
    // output: [B, S, D, H, W]
    auto output = at::zeros({B, S, D, H, W}, input.options());
    AT_DISPATCH_FLOATING_TYPES(input.type(), "img2sphere_cuda_forward", ([&] {
        img2sphere_cuda_forward(at::cuda::getCurrentCUDAStream(),
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    gamma.data<scalar_t>(),
                    planes.data<scalar_t>(),
                    planes_d.data<scalar_t>(),
                    B, 
                    H, W, 
                    D, S,
                    side_flag
                    );

    }));

    // std::cout <<"output" << std::endl;
    // output = output.contiguous().view({B, channel, depth_size, sphere_size});
    return output;
}



at::Tensor
// std::vector<at::Tensor>
sphere_cuda_backward(
            const at::Tensor &grad_output, 
            const at::Tensor &gamma,
            const at::Tensor &planes,
            const at::Tensor &planes_d,
            const float side_flag
            )
{


    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");
    AT_ASSERTM(gamma.is_contiguous(), "gamma tensor has to be contiguous");
    AT_ASSERTM(planes.is_contiguous(), "planes tensor has to be contiguous");
    AT_ASSERTM(planes_d.is_contiguous(), "planes_d tensor has to be contiguous");


    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");
    AT_ASSERTM(gamma.type().is_cuda(), "gamma must be a CUDA tensor");
    AT_ASSERTM(planes.type().is_cuda(), "planes must be a CUDA tensor");
    AT_ASSERTM(planes_d.type().is_cuda(), "planes_d must be a CUDA tensor");


    // output: [B, S, D, H, W]
    const int B = grad_output.size(0);
    const int S = grad_output.size(1);
    const int D = grad_output.size(2);
    const int H = grad_output.size(3);
    const int W = grad_output.size(4);
    // const int D = gamma.size(0);
    const int HW = H*W;

    AT_ASSERTM(planes_d.size(1)==planes.size(1) && planes_d.size(0)==planes.size(0),
        "(planes_d.shape!=planes.shape): (%d x %d) vs (%d x %d).", planes.size(0), planes_d.size(0), planes.size(1), planes_d.size(1));

    auto grad_input = at::zeros({B, HW, HW}, grad_output.options());
    // std::cout <<"output" <<grad_output << std::endl;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "img2sphere_cuda_backward", ([&] {
        img2sphere_cuda_backward(at::cuda::getCurrentCUDAStream(),
                    grad_input.data<scalar_t>(),
                    grad_output.data<scalar_t>(),
                    gamma.data<scalar_t>(),
                    planes.data<scalar_t>(),
                    planes_d.data<scalar_t>(),
                    B,
                    H, W, 
                    D, S, 
                    side_flag
                );

    }));

    return grad_input; 
}
