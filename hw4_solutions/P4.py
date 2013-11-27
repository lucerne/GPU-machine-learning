import numpy as np
import matplotlib.image as img

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
from pycuda.reduction import ReductionKernel

# Image files
in_file_name  = "Harvard_Tiny.png"
out_file_name = "Harvard_Sharpened_CPU.png"
# Sharpening constant
EPSILON    = np.float32(.005)

sharpen_source = \
"""
/** Helper macros for textual replacement **/
#define curr(i,j) curr[(j)+(i)*width]
#define next(i,j) next[(j)+(i)*width]

__global__ void sharpen(float* curr, float* next, float EPSILON,
                        int width, int height)
{
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if(0 < i && i < height-1 && 0 < j && j < width-1) {
    next(i,j) = curr(i,j) + EPSILON * (
                      -1*curr(i-1,j-1) + -2*curr(i-1,j) + -1*curr(i-1,j+1)
                    + -2*curr(i  ,j-1) + 12*curr(i  ,j) + -2*curr(i  ,j+1)
                    + -1*curr(i+1,j-1) + -2*curr(i+1,j) + -1*curr(i+1,j+1) );
  }
}
"""

# Compile sharpening kernel
module = nvcc.SourceModule(sharpen_source)
sharpen_kernel = module.get_function("sharpen")

# Compile variance kernel
var_kernel = ReductionKernel(dtype_out=np.float64, neutral="0",
                             reduce_expr="a+b", map_expr="(d[i]-mu)*(d[i]-mu)",
                             arguments="float* d, double mu")

def mean_variance(d_data):
  '''Return the mean and variance of a pycuda gpuarray'''
  mean = gpu.sum(d_data, dtype=np.float64).get() / d_data.size
  variance = var_kernel(d_data, mean).get() / d_data.size
  return mean, variance

if __name__ == '__main__':
  # Read image. BW images have R=G=B so extract the R-value
  # This is an np.array of np.float32s between 0.0 and 1.0
  image = np.array(img.imread(in_file_name)[:,:,0], dtype=np.float32)

  # Get image data
  height, width = np.int32(image.shape)
  print "Processing %d x %d image of %d %s pixels" % (width, height, image.size, image.dtype)

  # Define the block and grid size to use for the sharpening kernel
  block = (32,32,1)
  grid  = (int(np.ceil(float(width)/block[0])),
           int(np.ceil(float(height)/block[1])))

  # Transfer to the GPU
  d_curr = gpu.to_gpu(image)
  #d_next = gpu.to_gpu(image)    # Slow
  #d_next = d_curr.copy()        # Fast, PyCUDA v2013.1
  d_next = gpu.empty_like(d_curr)
  cu.memcpy_dtod(d_next.gpudata, d_curr.gpudata, d_curr.nbytes)


  # Compute the image's initial mean and variance
  init_mean, init_variance = mean_variance(d_curr)
  variance = 0

  # Perform repeated sharpening operations on the GPU
  while variance < 1.2 * init_variance:
    # Compute the sharpening
    sharpen_kernel(d_curr, d_next, EPSILON, width, height, block=block, grid=grid)

    # Swap references to the images, d_next => d_curr
    d_curr, d_next = d_next, d_curr

    # Compute mean and variance
    mean, variance = mean_variance(d_curr)
    print "Mean = %f,  Variance = %f" % (mean, variance)

  img.imsave('Harvard_Sharpened_GPU', d_curr.get(), cmap='gray', vmin=0, vmax=1)
