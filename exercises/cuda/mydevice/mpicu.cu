// --- CSCS (Swiss National Supercomputing Center) ---

#include <stdio.h>

extern "C"
void set_gpu(int dev)
{
  cudaSetDevice(dev);
}

extern "C"
void get_gpu_info(char *gpu_string, int dev)
{
  struct cudaDeviceProp dprop;
  cudaGetDeviceProperties(&dprop, dev);
  strcpy(gpu_string,dprop.name);
}

extern "C"
void get_more_gpu_info(int dev)
{
  int driverVersion = 0, runtimeVersion = 0;
  struct cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  printf("Device %d: \"%s\"\n", dev, deviceProp.name);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version     %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

  printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);

  printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

  printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
                  deviceProp.maxThreadsDim[0],
                  deviceProp.maxThreadsDim[1],
                  deviceProp.maxThreadsDim[2]);
}

