import time
import numpy as np
import matplotlib.image as img

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.scan as scan

#################################################
# GPU Grow ( queue based method )
# Use scan and directional growing

seed_source = """
__global__ void seed_image(float* img, int width, int height,
                            char* grow, int* front,
                            float seed_thresh_min, float seed_thresh_max)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = i * width + j;

  // If this is a valid pixel index
  if (0 <= k && k < width*height) {
    // Write 1 if it's in the seed threshold, 0 otherwise
    bool seed = (seed_thresh_min <= img[k] && img[k] <= seed_thresh_max);
    grow[k] = seed;
    front[k] = seed;
  }
}
"""

grow_source = """
__global__ void grow_image(float* img, int width, int height,
                            char* grow, int direction_i, int direction_j,
                            int* front_in, int front_size, int* front_out,
                            float grow_thresh_min, float grow_thresh_max)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= front_size)
    return;

  int k = front_in[gid];
  int i = k / width;
  int j = k % width;

  // If this is a valid pixel (we know it's in the region)
  if (k < width*height) {

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
        front_out[k] = 1;     // Set this pixel in the front (other directions)
      } else {
        return;               // Else, done
      }
    }
  }
}
"""

make_queue_source = """
__global__ void make_queue(int* front_scan, int* front_queue, int N)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid == 0) {
    if (front_scan[0] == 1)
      front_queue[0] = 0;
  } else if (gid < N) {
    if (front_scan[gid] > front_scan[gid-1]) {
      front_queue[front_scan[gid]-1] = gid;
    }
  }
}
"""


if __name__ == '__main__':
  in_file_name  = "Harvard_Huge.png"
  out_file_name = "Harvard_GrowRegion_GPU_B_Extra2.png"

  # Region growing constants [min, max]
  seed_threshold = np.float32([0, 0.08]);
  grow_threshold = np.float32([0, 0.27]);

  # Read image. BW images have R=G=B so extract the R-value
  image = np.array(img.imread(in_file_name)[:,:,0], dtype=np.float32)
  height, width = np.int32(image.shape)
  print "Processing %d x %d image" % (width, height)

  # Get the CUDA kernels
  seed_kernel = nvcc.SourceModule(seed_source).get_function("seed_image")
  grow_kernel = nvcc.SourceModule(grow_source).get_function("grow_image")
  compact_kernel = nvcc.SourceModule(make_queue_source).get_function("make_queue")
  scan_kernel = scan.InclusiveScanKernel(np.int32, "a+b")
  start_time = cu.Event()
  end_time = cu.Event()

  d2_block = (32,32,1)
  d2_grid = (int(np.ceil(float(width)/d2_block[0])),
             int(np.ceil(float(height)/d2_block[1])))

  d1_block = (512,1,1)
  d1_grid = (int(np.ceil(float(image.size)/d1_block[0])), 1)

  for test in range(10):

    start_time.record()

    # Initialize
    d_image = gpu.to_gpu(image)
    d_grow = gpu.empty(image.shape, dtype=np.int8)
    d_front_img = gpu.empty(image.size, dtype=np.int32)
    d_front_queue = gpu.zeros(image.size, dtype=np.int32)

    # Filter the seed kernels
    seed_kernel(d_image, width, height, d_grow, d_front_img,
                seed_threshold[0], seed_threshold[1],
                block=d2_block, grid=d2_grid)

    # Construct the front queue by compacting the front_img flags
    scan_kernel(d_front_img)
    front_size = np.int32(d_front_img[width*height-1:].get()[0])
    compact_kernel(d_front_img, d_front_queue, np.int32(image.size),
                   block=d1_block, grid=d1_grid)

    while front_size > 0:
      # Work out how many blocks we need for this front
      block = (512,1,1)
      grid = (int(np.ceil(float(front_size)/block[0])), 1)

      # Reset the front
      d_front_img.fill(np.int32(0))

      # For each direction
      for di,dj in np.int32([(1,0), (0,1), (-1,0), (0,-1)]):
        # Grow the region in the current direction
        grow_kernel(d_image, width, height,
                    d_grow, di, dj,
                    d_front_queue, front_size, d_front_img,
                    grow_threshold[0], grow_threshold[1],
                    block=block, grid=grid)

      # Construct the front queue by compacting the front_img flags
      scan_kernel(d_front_img)
      front_size = np.int32(d_front_img[width*height-1:].get()[0])
      compact_kernel(d_front_img, d_front_queue, np.int32(image.size),
                     block=d1_block, grid=d1_grid)

    end_time.record()
    end_time.synchronize()
    print "GPU time: %f" % (start_time.time_till(end_time) * 1e-3)

  img.imsave(out_file_name, d_grow.get(), cmap='gray')

