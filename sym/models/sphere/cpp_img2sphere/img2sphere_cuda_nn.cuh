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


// bilinear interpolation
template <typename scalar_t>
__device__ scalar_t bilinear_interpolation(const scalar_t *bottom_data, const int height, const int width, int offset_bc, scalar_t h, scalar_t w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh=1 - lh;
  scalar_t hw = 1 - lw;

  scalar_t v1 = bottom_data[offset_bc + h_low * width + w_low];
  scalar_t v2 = bottom_data[offset_bc + h_low * width + w_high];
  scalar_t v3 = bottom_data[offset_bc + h_high * width + w_low];
  scalar_t v4 = bottom_data[offset_bc + h_high * width + w_high];

  scalar_t w1 = hh * hw;
  scalar_t w2 = hh * lw; 
  scalar_t w3 = lh * hw; 
  scalar_t w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

// bilinear interpolation weight, useful for backpropagation
template <typename scalar_t>
__device__ void bilinear_interpolation_weight(const int height, const int width, scalar_t h, scalar_t w, scalar_t weights[])
{
  int h_low = floor(h);
  int w_low = floor(w);
  // int h_high = h_low + 1;
  // int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh=1 - lh;
  scalar_t hw = 1 - lw;

  // points outside the image already removed
  weights[0] = hh * hw;
  weights[1] = hh * lw;
  weights[2] = lh * hw;
  weights[3] = lh * lw;

}


// index_upper_triangle for making pairs
__forceinline__ __device__
int index_upper_t(int i, int j, int n)
{
  // w/o diagnoal
  // return (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1;
  // with diagnoal
  return k = n*(n-1)/2 - (n-i)*(n-i-1)/2 + j;
}

template <typename scalar_t>
__global__ void img2sphere_cuda_forward_kernel(const int n,
                                  const scalar_t *data_img,
                                  scalar_t *data_sphere,
                                  const scalar_t *gamma,
                                  const scalar_t *S,
                                  const scalar_t *P,
                                  const int batch, const int channel,
                                  const int height, const int width,
                                  const int depth, const int sphere
                                )
{
  scalar_t height_ = scalar_t(height);
  scalar_t width_ = scalar_t(width);
  const int pair = (height*width)*(height*width-1)/2;

  // launch height * width *depth * sphere cores
  CUDA_KERNEL_LOOP(index, n)
  {
    int h = index / width / depth / sphere % height; // h ind
    int w = index / depth / sphere % width; // w ind
    int d = index / sphere % depth; // depth ind
    int s = index % sphere; // sphere ind

    scalar_t y = 1 - (h+0.5) * 2.0 / height_;
    scalar_t x = (w+0.5) * 2.0 / width_ -1.0;

    scalar_t z = gamma[d];
    // printf("s=%5d, h=%3d, w=%3d, x=%.5f, y=%.5f, z=%.5f\n", s, h, w, x, y, z);

    // compute correspondence
    // S: [batch, sphere, 4, 4]
    // P:[batch, sphere, 4]
    for (int b=0; b < batch; ++b)
    { 
      int offset_S = b * sphere * 16 + s * 16; 
      int offset_P = b * sphere * 4 + s *4;
      scalar_t dis = P[offset_P] * x *z + P[offset_P+1] * y * z + P[offset_P+2] *z + P[offset_P+3] * 1.0;
      
      if (dis>0.0)
      {
        scalar_t xx = S[offset_S+0] * x * z + S[offset_S+1] * y * z + S[offset_S+2] * z + S[offset_S+3] * 1.0;
        scalar_t yy = S[offset_S+4] * x * z + S[offset_S+5] * y * z + S[offset_S+6] * z + S[offset_S+7] * 1.0;
        scalar_t zz = S[offset_S+8] * x * z + S[offset_S+9] * y * z + S[offset_S+10] * z + S[offset_S+11] * 1.0;
        scalar_t ee = S[offset_S+12] * x * z + S[offset_S+13] * y * z + S[offset_S+14] * z + S[offset_S+15] * 1.0;
        // printf("xx=%.5f, yy=%.5f, zz=%.5f, ee=%.5f \n", xx, yy, zz, ee);
      
        if(zz*ee<1e-6)
        {
          xx /= 1e-6;
          yy /= 1e-6;
        }
        else
        {
          xx /= zz*ee;
          yy /= zz*ee;
        }

        int ww = static_cast<int>(round((xx+1.0) * width_/2.0 - 0.5));
        int hh = static_cast<int>(round((1.0-yy) * height_/2.0 - 0.5));
        // printf("hh=%d, ww=%d, xx=%.5f, yy=%.5f \n", hh, ww, xx, yy);
        
        if( hh>=0 && hh <= height-1 && ww>=0 && ww <= width-1 )
        {
          // int h_low = floor(hh);
          // int w_low = floor(ww);
          // int h_high = h_low + 1;
          // int w_high = w_low + 1;

          // scalar_t weights[4]={0,0,0,0};
          // bilinear_interpolation_weight(height, width, hh, ww, weights);

          // int offset_bc=b*(channel*height*width)+c*(height*width);
          // int offset0 = offset_bc + h_low * width + w_low;
          // int offset1 = offset_bc + h_low * width + w_high;
          // int offset2 = offset_bc + h_high * width + w_low;
          // int offset3 = offset_bc + h_high * width + w_high;
          
          // scalar_t value_correspondence = weights[0]*data_img[offset0] + weights[1]*data_img[offset1] + weights[2]*data_img[offset2] + weights[3]*data_img[offset3];
          // printf("b=%3d, c=%3d, value=%.5f, value_correspondence=%.5f \n", b, c, value, value_correspondence);
          if (h * width + w<hh*width+ww) int p = index_upper_t(h * width + w, hh*width+ww, height*width);
          else int p = index_upper_t(hh*width+ww, h * width + w, height*width);
          // printf("b=%3d, p=%d \n", b, p);
          
          for (int c=0; c<channel; ++c)
          {
            // printf("b=%3d, c=%3d, p=%d, pair=%d \n", b, c, p, pair);
            int offset_img = b*channel*pair+c*pair+p;
            int offset_sphere = b*channel*sphere + c*sphere + s;
            scalar_t value = data_img[offset_img];
            // printf("b=%3d, c=%3d, p=%d, pair=%d, offset_img=%d, offset_sphere=%d, value=%f \n", b, c, p, pair, offset_img, offset_sphere);
            // if (offset_sphere>=batch*channel*sphere || offset_sphere<0)
            // {
            //   printf("ERROR: offset_sphere!: offset_sphere=%d, b=%d, c=%d, s=%d \n", offset_sphere, b, c, s);
            // }
            data_sphere[offset_sphere] += value;
            // atomicAdd(data_sphere+offset_sphere, value);
          }
        }
      }
    }
  }
}




template <typename scalar_t>
__global__ void img2sphere_cuda_backward_kernel(const int n,
                                    scalar_t* grad_img,
                                    const scalar_t* grad_sphere, 
                                    const scalar_t* gamma,
                                    const scalar_t* S,
                                    const scalar_t* P,
                                    const int batch, const int channel,
                                    const int height, const int width, 
                                    const int depth, const int sphere
                                  )
{
  scalar_t height_ = scalar_t(height);
  scalar_t width_ = scalar_t(width);
  const int pair = (height*width)*(height*width-1)/2;

  // launch height * width * depth * sphere cores
  CUDA_KERNEL_LOOP(index, n)
  {
    int h = index / width / depth / sphere % height; // h ind
    int w = index / depth / sphere % width; // w ind
    int d = index / sphere % depth; // depth ind
    int s = index % sphere; // sphere ind

    scalar_t y = 1.0 - (h+0.5) * 2.0 / height_;
    scalar_t x = (w+0.5) * 2.0 / width_ - 1.0;
    scalar_t z = gamma[d];
    // printf("s=%5d, h=%3d, w=%3d, x=%3d, y=%3d, x=%.5f\n", s, h, w, x, y, z);
    
    // compute correspondence
    // S: [batch, sphere, 4, 4]
    for (int b=0; b < batch; ++b)
    {
      int offset_S = b*sphere * 16 + s * 16; 
      int offset_P = b*sphere * 4 + s *4;
      scalar_t dis = P[offset_P] * x *z + P[offset_P+1] * y * z + P[offset_P+2] *z + P[offset_P+3] * 1.0;
      
      if (dis>0.0)
      {
        scalar_t xx = S[offset_S+0] * x * z + S[offset_S+1] * y * z + S[offset_S+2] * z + S[offset_S+3] * 1.0;
        scalar_t yy = S[offset_S+4] * x * z + S[offset_S+5] * y * z + S[offset_S+6] * z + S[offset_S+7] * 1.0;
        scalar_t zz = S[offset_S+8] * x * z + S[offset_S+9] * y * z + S[offset_S+10] * z + S[offset_S+11] * 1.0;
        scalar_t ee = S[offset_S+12] * x * z + S[offset_S+13] * y * z + S[offset_S+14] * z + S[offset_S+15] * 1.0;
        // printf("xx=%.5f, yy=%.5f, zz=%.5f, ee=%.5f \n", xx, yy, zz, ee);
        
        if(zz*ee<1e-6)
        {
          xx /= 1e-6;
          yy /= 1e-6;
        }
        else
        {
          xx /= zz*ee;
          yy /= zz*ee;
        }

        int ww= static_cast<int>(round((xx+1.0) * width_/2.0 - 0.5));
        int hh= static_cast<int>(round((1.0-yy) * height_/2.0 - 0.5));
        // printf("hh=%.5f, ww=%.5f, xx=%.5f, yy=%.5f, zz=%.5f \n", hh, ww, xx, yy, zz);
    
        if( hh>=0 && hh <= height-1 && ww>=0 && ww <= width-1 )
        {
          // int h_low = floor(hh);
          // int w_low = floor(ww);
          // int h_high = h_low + 1;
          // int w_high = w_low + 1;
          // scalar_t weights[4]={0,0,0,0};
          // bilinear_interpolation_weight(height, width, hh, ww, weights);

          // int offset_bc=b*(channel*height*width)+c*(height*width);
          // int offset0 = offset_bc + h_low * width + w_low;
          // int offset1 = offset_bc + h_low * width + w_high;
          // int offset2 = offset_bc + h_high * width + w_low;
          // int offset3 = offset_bc + h_high * width + w_high;

          // // add grad_value to the sphere point
          // atomicAdd(grad_img+offset0, weights[0]*grad_value);
          // atomicAdd(grad_img+offset1, weights[1]*grad_value);
          // atomicAdd(grad_img+offset2, weights[2]*grad_value);
          // atomicAdd(grad_img+offset3, weights[3]*grad_value);

          // printf("b=%3d, c=%3d, h_low=%3d, w_low=%3d, hh=%.3f, ww=%.3f, weight=%.5f, grad_value=%.5f \n", b, c, h_low, w_low, hh, ww, weights[0], grad_value);
          // printf("b=%3d, c=%3d, h_low=%3d, w_high=%3d, hh=%.3f, ww=%.3f, weight=%.5f, grad_value=%.5f \n", b, c, h_low, w_high, hh, ww, weights[1], grad_value);
          // printf("b=%3d, c=%3d, h_high=%3d, w_low=%3d, hh=%.3f, ww=%.3f, weight=%.5f, grad_value=%.5f \n", b, c, h_high, w_low, hh, ww, weights[2], grad_value);
          // printf("b=%3d, c=%3d, h_high=%3d, w_high=%3d, hh=%.3f, ww=%.3f, weight=%.5f, grad_value=%.5f \n", b, c, h_high, w_high, hh, ww, weights[3], grad_value);
          
          int p = index_upper_t(h * width + w, hh*width+ww, height*width);
          
          // grad_sphere: [batch, channel, sphere]
          // grad_img: [batch, channel, height, width]
          for (int c=0; c<channel; ++c)
          {
            int offset_sphere =  b * (channel * sphere) + c * sphere + s;
            scalar_t grad_value = grad_sphere[offset_sphere];
            // printf("b=%5d, c=%5d, s=%5d, offset_sphere=%5d, grad=%.5f, d=%5d, h=%5d, w=%5d \n", b, c, s, offset_sphere, grad_value, d, h, w);
            int offset_img = b*channel*pair+c*pair+p;
            atomicAdd(grad_img+offset_img, grad_value);
          }
        }
      }
    }
  }
}



template <typename scalar_t>
void img2sphere_cuda_forward(cudaStream_t stream,
                  const scalar_t* data_img, 
                  scalar_t* data_sphere,
                  const scalar_t* gamma,
                  const scalar_t* S,
                  const scalar_t* P,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int depth, const int sphere
                ) 
{
  const int num_kernels = height * width * depth * sphere;
  // printf("num_kernels=%d, height=%d, width=%d, depth=%d, sphere=%d \n", num_kernels, height, width, depth, sphere);
  // printf("CUDA_NUM_THREADS: CUDA_NUM_THREADS=%d \n", CUDA_NUM_THREADS);
  img2sphere_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          data_img, 
                                                          data_sphere,
                                                          gamma, 
                                                          S,
                                                          P,
                                                          batch, channel, 
                                                          height, width, 
                                                          depth, sphere
                                                        );
                                                      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in img2sphere_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void img2sphere_cuda_backward(cudaStream_t stream,
                  scalar_t* grad_img,
                  const scalar_t* grad_sphere, 
                  const scalar_t* gamma,
                  const scalar_t* S,
                  const scalar_t* P,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int depth, const int sphere
                  )
{

  const int num_kernels = height * width * depth * sphere;
  // ***********************************************************************//
  img2sphere_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_img, 
                                                          grad_sphere, 
                                                          gamma,
                                                          S,
                                                          P,
                                                          batch, channel, 
                                                          height, width,
                                                          depth, sphere
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in im2sphere_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}


