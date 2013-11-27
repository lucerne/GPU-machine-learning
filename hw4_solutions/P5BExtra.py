import time
import numpy as np
import matplotlib.image as img

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu

#################################################
# GPU Grow ( queue based method )

source_string = """
__device__ bool in_range(float p, float min, float max) {
  return min <= p && p <= max;
}

__global__ void seed2_image(float* img, int width, int height,
                            int* front_out,
                            float seed_thresh_min, float seed_thresh_max)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = i * width + j;

  // If this is a valid pixel index
  if (0 <= k && k < width*height) {
    // If this pixel should be in the region
    if (in_range(img[k], seed_thresh_min, seed_thresh_max)) {
      // Insert it into the pixel front
      front_out[k] = k;
    }
  }
}

__global__ void grow2_image(float* img, int width, int height,
                            int* front_in, int front_in_length,
                            char* grow, int* front_out,
                            float grow_thresh_min, float grow_thresh_max)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (0 <= gid && gid < front_in_length) {
    int k = front_in[gid];
    int i = k / width;
    int j = k % width;

    // Flag this pixel as in the grow region
    grow[k] = 1;

    // Add neighbors to the front if they are to be added to the region
    if (i > 0) {
      int k2 = k - width;
      if (grow[k2] == 0 && in_range(img[k2], grow_thresh_min, grow_thresh_max))
        front_out[gid + 0*front_in_length] = k2;
    }
    if (i < height-1) {
      int k2 = k + width;
      if (grow[k2] == 0 && in_range(img[k2], grow_thresh_min, grow_thresh_max))
        front_out[gid + 1*front_in_length] = k2;
    }
    if (j > 0) {
      int k2 = k - 1;
      if (grow[k2] == 0 && in_range(img[k2], grow_thresh_min, grow_thresh_max))
        front_out[gid + 2*front_in_length] = k2;
    }
    if (j < width-1) {
      int k2 = k + 1;
      if (grow[k2] == 0 && in_range(img[k2], grow_thresh_min, grow_thresh_max))
        front_out[gid + 3*front_in_length] = k2;
    }
  }
}
"""

if __name__ == '__main__':
  in_file_name  = "Harvard_Huge.png"
  out_file_name = "Harvard_GrowRegion_GPU_B_Extra.png"

  # Region growing constants [min, max]
  seed_threshold = np.float32([0, 0.08]);
  grow_threshold = np.float32([0, 0.27]);

  # Read image. BW images have R=G=B so extract the R-value
  image = np.array(img.imread(in_file_name)[:,:,0], dtype=np.float32)
  height, width = np.int32(image.shape)
  print "Processing %d x %d image" % (width, height)

  # Get the CUDA kernels
  module = nvcc.SourceModule(source_string)
  seed_kernel = module.get_function("seed2_image")
  grow_kernel = module.get_function("grow2_image")
  start_time = cu.Event()
  end_time = cu.Event()

  for test in range(10):

    block = (32,32,1)
    grid = (int(np.ceil(float(width)/block[0])),
            int(np.ceil(float(height)/block[1])))

    start_time.record()

    d_image = gpu.to_gpu(image)
    d_grow  = gpu.zeros(image.shape, dtype=np.int8)
    d_next_front = gpu.empty(image.shape, dtype=np.int32)
    d_next_front.fill(np.int32(-1))

    seed_kernel(d_image, width, height,
                d_next_front,
                seed_threshold[0], seed_threshold[1],
                block=block, grid=grid)

    # Make the pixel front by extracting the non-negative-one values
    next_front = d_next_front.get()
    pixel_front = next_front[next_front != -1]

    while pixel_front.size > 0:
      # Define the block and grid size
      block = (512,1,1)
      grid = (int(np.ceil(float(pixel_front.size)/block[0])), 1)

      # Pass the front to the GPU
      d_pixel_front = gpu.to_gpu(pixel_front)

      # Initialize the next_front, making sure there is enough space
      d_next_front = gpu.empty(4*d_pixel_front.size, dtype=np.int32)
      d_next_front.fill(np.int32(-1))

      grow_kernel(d_image, width, height,
                  d_pixel_front, np.int32(pixel_front.size),
                  d_grow, d_next_front,
                  grow_threshold[0], grow_threshold[1],
                  block=block, grid=grid)

      # Make the new front queue
      next_front = d_next_front.get()
      pixel_front = np.unique(next_front[next_front != -1])

    end_time.record()
    end_time.synchronize()
    print "GPU time: %f" % (start_time.time_till(end_time) * 1e-3)

  img.imsave(out_file_name, d_grow.get(), cmap='gray')

