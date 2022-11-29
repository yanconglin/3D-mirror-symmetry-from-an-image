#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#include "cuda_utils.h"


// index_upper_triangle for making pairs
__forceinline__ __device__
int index_upper_t(int i, int j, int n)
{ 
  // w/o diagnoal
  // https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
  // return (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1;
  // with diagnoal
  // https://math.stackexchange.com/questions/2134011/conversion-of-upper-triangle-linear-index-from-index-on-symmetrical-array
  return n*(n-1)/2 - (n-i)*(n-i-1)/2 + j;
}

static __forceinline__ __device__
bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__
bool within_bounds_2d_(float h, float w, float H, float W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void img2sphere_cuda_forward_kernel(const int n,
                                  const scalar_t *atten,
                                  scalar_t *sphere,
                                  const scalar_t *gamma,
                                  const scalar_t *plane,
                                  const scalar_t *plane_d,
                                  const int B, 
                                  const int H, const int W,
                                  const int D, const int S,
                                  const float side_flag
                                )
{
  const scalar_t H_ = static_cast<scalar_t>(H);
  const scalar_t W_ = static_cast<scalar_t>(W);
  const int HW = H*W;

  // launch B * H * W *D * S cores
  // atten: [B, HW, HW]
  // sphere:[B, S, D, HW]
  CUDA_KERNEL_LOOP(index, n)
  {
    int b = (index / H / W / D / S) % B; // b ind
    int h = (index / W / D / S) % H; // h ind
    int w = (index / D / S) % W; // w ind
    int d = (index / S) % D; // depth ind
    int s = index % S; // sphere ind

    scalar_t y = 1.0 - (h+0.5) * 2.0 / H_;
    scalar_t x = (w+0.5) * 2.0 / W_ -1.0;
    scalar_t z = gamma[d];
    // printf("s=%d, h=%d, w=%d, x=%.6f, y=%.6f, z=%.6f, height=%d, width=%d \n", s, height, width, x, y, z, height, width);

    // compute correspondence
    // S: [B, S, 4, 4]
    int offset_p = b * S * 16 + s * 16; 
    int offset_d = b * S * 4 + s *4;
    scalar_t dis = plane_d[offset_d] * x *z + plane_d[offset_d+1] * y * z + plane_d[offset_d+2] *z + plane_d[offset_d+3];
    
    if (dis<side_flag) // far side
    {
      scalar_t xx = plane[offset_p+0] * x * z + plane[offset_p+1] * y * z + plane[offset_p+2] * z + plane[offset_p+3];
      scalar_t yy = plane[offset_p+4] * x * z + plane[offset_p+5] * y * z + plane[offset_p+6] * z + plane[offset_p+7];
      scalar_t zz = plane[offset_p+8] * x * z + plane[offset_p+9] * y * z + plane[offset_p+10] * z + plane[offset_p+11];
      // scalar_t ee = plane[offset_p+12] * x * z + plane[offset_p+13] * y * z + plane[offset_p+14] * z + plane[ofoffset_pfset_S+15];
      // printf("xx=%.6f, yy=%.6f, zz=%.6f, height=%d, width=%d \n", xx, yy, zz, height, width);
    
      if(zz>1e-6)
      {
        xx /= zz;
        yy /= zz;

        scalar_t ww = (xx+1.0) * W_ / 2.0 - 0.5;
        scalar_t hh = (1.0-yy) * H_ / 2.0 - 0.5;
        // printf("zz>1e-6: b=%d, s=%d, h=%d, w=%d, d=%d, x=%.6f, y=%.6f, hh=%f, ww=%f, xx=%.6f, yy=%.6f, height=%d, width=%d \n", b, s, height, width, d, x, y, hheight, widthw, xx, yy, height, width);
        
        if (within_bounds_2d_(hh, ww, H_, W_))
        { 
          // printf("within_bounds_2d_: b=%d, s=%d, h=%d, w=%d, d=%d, hh=%f, ww=%f, xx=%.6f, yy=%.6f, height=%d, width=%d \n", b, s, height, width, d, hheight, widthw, xx, yy, height, width);

          int hh_nw = static_cast<int>(floor(hh));
          int ww_nw = static_cast<int>(floor(ww));
          int hh_ne = hh_nw;
          int ww_ne = ww_nw + 1;
          int hh_sw = hh_nw + 1;
          int ww_sw = ww_nw;
          int hh_se = hh_nw + 1;
          int ww_se = ww_nw + 1;

          // get surfaces to each neighbor:
          scalar_t weight_nw = (hh_se - hh) * (ww_se - ww);
          scalar_t weight_ne = (hh_sw - hh) * (ww - ww_sw);
          scalar_t weight_sw = (ww_ne - ww) * (hh - hh_ne);
          scalar_t weight_se = (ww - ww_nw) * (hh - hh_nw);
          
          int hw = h * W + w;
          // int hhww = hh * W + ww;
          int nw = hh_nw*W+ww_nw;
          int ne = hh_ne*W+ww_ne;
          int sw = hh_sw*W+ww_sw;
          int se = hh_se*W+ww_se;

          // printf("within_bounds_2d_: b=%d, s=%d, h=%d, w=%d, d=%d, hh=%f, ww=%f, xx=%.6f, yy=%.6f, \
                  weight_nw=%.6f, weight_ne=%.6f, weight_sw=%.6f, weight_se=%.6f \n", \
                  b, s, height, width, d, hheight, widthw, xx, yy, weight_nw, weight_ne, weight_sw, weight_se);

          // atten: [B, HW, HW]
          // sphere: [B, S, D, H, w]
          scalar_t v_cor = 0.0;
          int offset_atten = b*HW*HW;
          if (within_bounds_2d(hh_nw, ww_nw, H, W))
          {
            v_cor += weight_nw * atten[offset_atten+hw*HW+nw];
          }
  
          if (within_bounds_2d(hh_ne, ww_ne, H, W))
          {
            v_cor += weight_ne * atten[offset_atten+hw*HW+ne];
          }
  
          if (within_bounds_2d(hh_sw, ww_sw, H, W))
          {
            v_cor += weight_sw * atten[offset_atten+hw*HW+sw];
          }
  
          if (within_bounds_2d(hh_se, ww_se, H, W))
          {
            v_cor += weight_se * atten[offset_atten+hw*HW+se];
          }

          // sphere: [B, S, D, HW]
          int offset_sphere = b*S*D*HW + s*D*HW + d*HW+ hw;
          atomicAdd(sphere+offset_sphere, v_cor);

        } 
      }
    }
  }
}


template <typename scalar_t>
__global__ void img2sphere_cuda_backward_kernel(const int n,
                                    scalar_t* grad_atten,
                                    const scalar_t* grad_sphere, 
                                    const scalar_t* gamma,
                                    const scalar_t *plane,
                                    const scalar_t *plane_d,
                                    const int B, 
                                    const int H, const int W,
                                    const int D, const int S,
                                    const float side_flag
                                  )
{
  const scalar_t H_ = static_cast<scalar_t>(H);
  const scalar_t W_ = static_cast<scalar_t>(W);
  const int HW = H*W;

  // launch B * H * W *D * S cores
  // atten: [B, HW, HW]
  // sphere:[B, S, D, HW]
  CUDA_KERNEL_LOOP(index, n)
  {
    int b = (index / H / W / D / S) % B; // b ind
    int h = (index / W / D / S) % H; // h ind
    int w = (index / D / S) % W; // w ind
    int d = (index / S) % D; // depth ind
    int s = index % S; // sphere ind

    scalar_t y = 1.0 - (h+0.5) * 2.0 / H_;
    scalar_t x = (w+0.5) * 2.0 / W_ - 1.0;
    scalar_t z = gamma[d];
    // printf("s=%5d, h=%3d, w=%3d, x=%3d, y=%3d, x=%.5f\n", s, height, width, x, y, z);
    
    // compute correspondence
    // S: [B, S, 4, 4]
    int offset_p = b * S * 16 + s * 16; 
    int offset_d = b * S * 4 + s *4;
    scalar_t dis = plane_d[offset_d] * x *z + plane_d[offset_d+1] * y * z + plane_d[offset_d+2] *z + plane_d[offset_d+3];
    
    if (dis<side_flag) // far side
    {
      scalar_t xx = plane[offset_p+0] * x * z + plane[offset_p+1] * y * z + plane[offset_p+2] * z + plane[offset_p+3];
      scalar_t yy = plane[offset_p+4] * x * z + plane[offset_p+5] * y * z + plane[offset_p+6] * z + plane[offset_p+7];
      scalar_t zz = plane[offset_p+8] * x * z + plane[offset_p+9] * y * z + plane[offset_p+10] * z + plane[offset_p+11];
      // scalar_t ee = plane[offset_p+12] * x * z + plane[offset_p+13] * y * z + plane[offset_p+14] * z + plane[offset_p+15];
      // printf("xx=%.6f, yy=%.6f, zz=%.6f, height=%d, width=%d \n", xx, yy, zz, height, width);
    
      if(zz>1e-6)
      {
        xx /= zz;
        yy /= zz;

        scalar_t ww = (xx+1.0) * W_ /2.0 - 0.5;
        scalar_t hh = (1.0-yy) * H_ /2.0 - 0.5;
        // printf("hh=%d, ww=%d, xx=%.5f, yy=%.5f \n", hheight, widthw, xx, yy);
        
        if (within_bounds_2d_(hh, ww, H_, W_))
        { 
          int hh_nw = static_cast<int>(floor(hh));
          int ww_nw = static_cast<int>(floor(ww));
          int hh_ne = hh_nw;
          int ww_ne = ww_nw + 1;
          int hh_sw = hh_nw + 1;
          int ww_sw = ww_nw;
          int hh_se = hh_nw + 1;
          int ww_se = ww_nw + 1;

          // get surfaces to each neighbor:
          scalar_t weight_nw = (hh_se - hh) * (ww_se - ww);
          scalar_t weight_ne = (hh_sw - hh) * (ww - ww_sw);
          scalar_t weight_sw = (ww_ne - ww) * (hh - hh_ne);
          scalar_t weight_se = (ww - ww_nw) * (hh - hh_nw);
          
          int hw = h * W + w;
          // int hhww = hh * W + ww;
          int nw = hh_nw*W+ww_nw;
          int ne = hh_ne*W+ww_ne;
          int sw = hh_sw*W+ww_sw;
          int se = hh_se*W+ww_se;

          // atten: [B, HW, HW]
          // sphere: [B, S, D, H, W]
          int offset_sphere = b*S*D*HW + s*D*HW + d*HW+ hw;
          scalar_t v_grad = grad_sphere[offset_sphere];

          int offset_atten = b*HW*HW+hw*HW;
          if (within_bounds_2d(hh_nw, ww_nw, H, W))
          {
            // grad to atten
            atomicAdd(grad_atten+offset_atten+nw, weight_nw*v_grad);
          }
  
          if (within_bounds_2d(hh_ne, ww_ne, H, W))
          {
            // grad to atten
            atomicAdd(grad_atten+offset_atten+ne, weight_ne*v_grad);          
          }
  
          if (within_bounds_2d(hh_sw, ww_sw, H, W))
          {
            // grad to atten
            atomicAdd(grad_atten+offset_atten+sw, weight_sw*v_grad);          
          }
  
          if (within_bounds_2d(hh_se, ww_se, H, W))
          {
            // grad to atten
            atomicAdd(grad_atten+offset_atten+se, weight_se*v_grad);                
          }
        }
      }
    }
  }
}



template <typename scalar_t>
void img2sphere_cuda_forward(cudaStream_t stream,
                  const scalar_t* atten, 
                  scalar_t* sphere,
                  const scalar_t* gamma,
                  const scalar_t* planes,
                  const scalar_t* planes_d,
                  const int B, 
                  const int H, const int W, 
                  const int D, const int S,
                  const int side_flag
                ) 
{
  const int num_kernels = B * H * W * D * S;
  // printf("img2sphere_cuda_forward num_kernels=%d, height=%d, width=%d, depth=%d, sphere=%d \n", num_kernels, height, width, depth, sphere);
  // printf("CUDA_NUM_THREADS: CUDA_NUM_THREADS=%d \n", CUDA_NUM_THREADS);
  img2sphere_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          atten,
                                                          sphere,
                                                          gamma, 
                                                          planes,
                                                          planes_d,
                                                          B, 
                                                          H, W, 
                                                          D, S, 
                                                          side_flag
                                                        );
                                                      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in img2sphere_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void img2sphere_cuda_backward(cudaStream_t stream,
                  scalar_t* grad_atten,
                  const scalar_t* grad_sphere, 
                  const scalar_t* gamma,
                  const scalar_t* planes,
                  const scalar_t* planes_d,
                  const int B, 
                  const int H, const int W, 
                  const int D, const int S,
                  const int side_flag
                  )
{

  const int num_kernels = B * H * W * D * S;
  // printf("backward num_kernels: B=%d, H=%d, W=%d, D=%d, S=%d, num_kernels=%d, CUDA_NUM_THREADS=%d \n", B, H, W, D, S, num_kernels. CUDA_NUM_THREADS);
  // ***********************************************************************//
  img2sphere_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_atten,
                                                          grad_sphere, 
                                                          gamma,
                                                          planes,
                                                          planes_d,
                                                          B, 
                                                          H, W,
                                                          D, S,
                                                          side_flag
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in im2sphere_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}




