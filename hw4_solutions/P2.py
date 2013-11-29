import numpy as np
import pylab as pl

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu

D2x_source = """
/** Second order central difference stencil
 *
 * Input: Array a of length N and grid size dx
 * Output: D2a[i] = (a[i-1] - 2*a[i] + a[i+1]) / (dx*dx) for all 0 < i < N-1
 */
__global__ void D2x_kernel(double* D2a, double* a, int N, double dx)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (0 < i && i < N-1 )
    D2a[i] = (a[i-1] - 2*a[i] + a[i+1]) / (dx*dx);
}
"""

if __name__ == '__main__':
  D2x_kernel = nvcc.SourceModule(D2x_source).get_function("D2x_kernel")

  N = np.int32(2**20)
  x = np.linspace(0, 1, N)
  a = np.sin(np.pi*x)
  dx = np.float64(1.0/float(N-1))

  d_a   = gpu.to_gpu(a)
  d_D2a = gpu.zeros_like(d_a)

  block = (512,1,1)
  grid = (int(np.ceil(float(N)/block[0])),1)

  D2x_kernel(d_D2a, d_a, N, dx, block=block, grid=grid)

  pl.plot(x, d_D2a.get())
  pl.show()
