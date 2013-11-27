import numpy as np

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu

minreduce_source = """
__global__ void min_kernel(float* d_data, int N, float* d_min)
{
  // Allocate shared memory in the kernel call
  extern __shared__ float s_data[];

  // Compute local and global thread id
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;

  // Copy data to shared memory
  if (gid < N)
    s_data[tid] = d_data[gid];
  else
    s_data[tid] = d_data[0];   // Pad values if we're the last thread block
  __syncthreads();

  // Reduce the minimum value into s_data[0]
  for (int n = blockDim.x / 2; n > 0 && tid < n; n /= 2) {
    if (tid < n && s_data[tid + n] < s_data[tid])
      s_data[tid] = s_data[tid + n];
    __syncthreads();
  }

  // Store the minimum value back to global memory
  if (tid == 0)
    d_min[blockIdx.x] = s_data[0];
}
"""

def gpu_reduce(kernel, d_a):
  while d_a.size > 1:
    # Array size
    N = np.int32(d_a.size)
    # Block size (threads per block)
    b_size = (512,1,1)
    # Grid size (blocks per grid)
    g_size = (int(np.ceil(float(N)/b_size[0])),1)
    # Shared memory size (bytes per block)
    s_size = int(b_size[0] * d_a.dtype.itemsize)

    # Allocate output array with one element per thread block
    d_min = gpu.empty(g_size[0], dtype = d_a.dtype)

    # Run the GPU kernel
    kernel(d_a, N, d_min, block=b_size, grid=g_size, shared=s_size)

    # Replace the array reference and iterate again if needed
    d_a = d_min

  # Return the (single) element of the final array
  return d_a.get()[0]


if __name__ == '__main__':
  # Compile the CUDA Kernel
  module = nvcc.SourceModule(minreduce_source)
  # Return a handle to the compiled CUDA kernel
  min_kernel = module.get_function("min_kernel")

  # Prepare a random array
  N = 2**20
  a = np.float32(np.random.random(N))
  d_a = gpu.to_gpu(a)

  # Print the GPU result
  print gpu_reduce(min_kernel, d_a)
  # Print the CPU result
  print np.min(a)
