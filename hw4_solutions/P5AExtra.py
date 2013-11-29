import time
import numpy as np
import matplotlib.image as img

import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.autoinit

#################################################
# GPU Grow ( full image method )

seed1_source = """
__global__ void seed1_image(float* img, int width, int height,
                            char* grow,
                            float seed_thresh_min, float seed_thresh_max)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = i * width + j;

  // If this is a valid pixel index
  if (0 <= k && k < width*height) {
    // Write 1 if it's in the seed threshold, 0 otherwise
    grow[k] = (seed_thresh_min <= img[k] && img[k] <= seed_thresh_max);
  }
}
"""

grow1_source = """
__global__ void grow1_image(float* img, int width, int height,
                            char* grow, int direction_i, int direction_j,
                            float grow_thresh_min, float grow_thresh_max)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = i * width + j;

  // If this is a valid pixel and it is in the region
  if (0 <= k && k < width*height && grow[k] == 1) {

    // Step 32 times in the specified direction
    for (int step = 0; step < 32; ++step) {
      i += direction_i;
      j += direction_j;
      k = i * width + j;

      // If this pixel is still in-bounds,
      //               is not in the region
      //           and should be in the region (already know a neighbor is)
      if (0 <= i && i < height && 0 <= j && j < width
          && grow[k] == 0     // This isn't a race condition!
          && grow_thresh_min < img[k] && img[k] < grow_thresh_max) {
        grow[k] = 1;          // Set this pixel in the region
      } else {
        return;               // Else, done
      }
    }
  }
}
"""


if __name__ == '__main__':
  in_file_name  = "Harvard_Huge.png"
  out_file_name = "Harvard_GrowRegion_GPU_A_Extra.png"

  # Region growing constants [min, max]
  seed_threshold = np.float32([0, 0.08]);
  grow_threshold = np.float32([0, 0.27]);

  # Read image. BW images have R=G=B so extract the R-value
  image = np.array(img.imread(in_file_name)[:,:,0], dtype=np.float32)
  height, width = np.int32(image.shape)
  print "Processing %d x %d image" % (width, height)

  # Get the CUDA kernels
  seed_kernel = nvcc.SourceModule(seed1_source).get_function("seed1_image")
  grow_kernel = nvcc.SourceModule(grow1_source).get_function("grow1_image")
  start_time = cu.Event()
  end_time = cu.Event()

  for test in range(10):

    # Define the block size and grid size
    block = (32,32,1)
    grid = (int(np.ceil(float(width)/block[0])),
            int(np.ceil(float(height)/block[1])))

    start_time.record()

    # Initialize the GPU image and the grow region
    d_image = gpu.to_gpu(image)
    d_grow  = gpu.zeros(image.shape, dtype=np.int8)

    # Find seed pixels
    seed_kernel(d_image, width, height, d_grow,
                seed_threshold[0], seed_threshold[1],
                block=block, grid=grid)

    # Define the number of the flagged pixels in the grow region
    grow_size_prev = 0
    grow_size = gpu.sum(d_grow, dtype=np.int32).get()

    while grow_size > grow_size_prev:

      # For each direction
      for di,dj in np.int32([(1,0), (0,1), (-1,0), (0,-1)]):
        # Grow the region in this direction
        grow_kernel(d_image, width, height,
                    d_grow, di, dj,
                    grow_threshold[0], grow_threshold[1],
                    block=block, grid=grid)

      # Find the number of pixels in the grow region
      grow_size_prev = grow_size
      grow_size = gpu.sum(d_grow, dtype=np.int32).get()

    end_time.record()
    end_time.synchronize()
    print "GPU time: %f" % (start_time.time_till(end_time) * 1e-3)

  img.imsave(out_file_name, d_grow.get(), cmap='gray')

