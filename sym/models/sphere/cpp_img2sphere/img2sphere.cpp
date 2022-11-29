#include "img2sphere.h"
#include "sphere_cuda.h"

// std::vector<at::Tensor>
at::Tensor
img2sphere_forward(
            const at::Tensor &input,
            const at::Tensor &gamma,
            const at::Tensor &planes,
            const at::Tensor &planes_d,
            const float side_flag
            )
{
    if (input.type().is_cuda())
    {
        return sphere_cuda_forward(
                                input, 
                                gamma,
                                planes,
                                planes_d,
                                side_flag
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor
// std::vector<at::Tensor>
img2sphere_backward(
                const at::Tensor &grad_output,
                const at::Tensor &gamma,
                const at::Tensor &planes,
                const at::Tensor &planes_d,
                const float side_flag
                )
{
    if (grad_output.type().is_cuda())
    {
        return sphere_cuda_backward(
                                grad_output, 
                                gamma,
                                planes,
                                planes_d,
                                side_flag
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("img2sphere_forward", &img2sphere_forward, "Forward pass of sphere");
    m.def("img2sphere_backward", &img2sphere_backward, "Backward pass of sphere");
}
