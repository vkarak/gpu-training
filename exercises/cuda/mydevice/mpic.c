// --- CSCS (Swiss National Supercomputing Center) ---

#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>

#ifndef DEVS_PER_NODE
#define DEVS_PER_NODE 1  // Devices per node
#endif

void set_gpu(int);
void get_gpu_info(char *, int);
void get_more_gpu_info(int);

int main(int argc, char *argv[])
{
    int rank=0, size=0, namelen;
    char gpu_str[256] = "";
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);  
    MPI_Comm_size (MPI_COMM_WORLD, &size);  
    MPI_Get_processor_name(processor_name, &namelen);
    int dev = rank % DEVS_PER_NODE;
    set_gpu(dev);

    // step1: cudaGetDeviceProperties
    get_gpu_info(gpu_str, dev);
    printf("=== get_gpu_info ===\n");
    printf("Process %d on %s out of %d Device %d (%s)\n", rank, processor_name, size, dev, gpu_str); 

    // step2: more cudaGetDeviceProperties
    printf("\n=== get_more_gpu_info ===\n");
    get_more_gpu_info(dev); 

    // step3: /proc/driver/nvidia
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n=== /proc/driver/nvidia/version ===\n");
    static const char filename[] = "/proc/driver/nvidia/version";
    FILE *file = fopen ( filename, "r" );
    if ( file != NULL ) {
        fgets ( gpu_str, sizeof gpu_str, file ) ;
        fputs ( gpu_str, stdout ); 
    }
    fclose(file);

    MPI_Finalize();

    return 0;
}

// srun -C gpu -t1 -n1 -c1 --hint=nomultithread ./exe.dom
// 
// === get_gpu_info ===
// Process 0 on nid00001 out of 1 Device 0 (Tesla P100-PCIE-16GB)
// 
// === get_more_gpu_info ===
// Device 0: "Tesla P100-PCIE-16GB"
//   CUDA Driver Version / Runtime Version     8.0 / 8.0
//   CUDA Capability Major/Minor version number:    6.0
//   (56) Multiprocessors, ( 64) CUDA Cores/MP:     3584 CUDA Cores
//   Maximum number of threads per multiprocessor:  2048
//   Peak number of threads:                        114688 threads
//   Maximum number of threads per block:           1024
// 
// === /proc/driver/nvidia/version ===
// NVRM version: NVIDIA UNIX x86_64 Kernel Module  375.39  Tue Jan 31 20:47:00 PST 2017
