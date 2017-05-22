// --- CSCS (Swiss National Supercomputing Center) ---
// samples/1_Utilities/deviceQuery/deviceQuery.cpp

#include <stdio.h>
#include <helper_cuda.h>

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
  printf("  CUDA Driver Version / Runtime Version     %d.%d / %d.%d\n", 
    driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, 
    (runtimeVersion%100)/10);

  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", 
    deviceProp.major, deviceProp.minor);

  printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
  deviceProp.multiProcessorCount,
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) 
                            * deviceProp.multiProcessorCount);

  printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", 
    deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
  printf("  Theoretical peak performance per GPU:          %.0f Gflop/s\n",
    deviceProp.clockRate *1e-6f 
    *_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
    *deviceProp.multiProcessorCount);

  printf("  Maximum number of threads per multiprocessor:  %d\n", 
    deviceProp.maxThreadsPerMultiProcessor);

  printf("  Peak number of threads:                        %d threads\n", 
    deviceProp.multiProcessorCount 
    *deviceProp.maxThreadsPerMultiProcessor );

  printf("  Maximum number of threads per block:           %d\n", 
    deviceProp.maxThreadsPerBlock);

}

