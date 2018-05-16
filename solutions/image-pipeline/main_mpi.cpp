/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <openacc.h>
#include <omp.h>
#include <mpi.h>

extern "C" {
void blur5(unsigned char*,unsigned char*,long,long,long,long,int,int);
void blur5_blocked(unsigned char*,unsigned char*,long,long,long,long);
void blur5_update(unsigned char*,unsigned char*,long,long,long,long,int,int);
void blur5_pipelined(unsigned char*,unsigned char*,long,long,long,long);
void blur5_pipelined_multi(unsigned char*,unsigned char*,long,long,long,long);
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  printf("DEBUG: world rank %d world size %d\n", world_rank, world_size);
  if ( (world_rank==0) && (argc < 3))
  {
    fprintf(stderr,"Usage: %s inFilename outFilename\n",argv[0]);
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  IplImage* img;
  unsigned char *data, *out, *work; 
  long width, height, ch, ws, sz,my_block_size;
  if (world_rank==0)
  {
    img = cvLoadImage(argv[1]);
    printf("%s: %d x %d, %d %d\n", argv[1],img->width, img->height, img->widthStep, img->nChannels);

    width = img->width;
    height = img->height;
    ch = img->nChannels;
    ws = img->widthStep;
    sz = height * ws;
    data = (unsigned char*)calloc((height + 4)*ws,sizeof(unsigned char));;
    memcpy(&data[2*ws], img->imageData, height*ws*sizeof(unsigned char));
    out = (unsigned char*)malloc(sz * sizeof(unsigned char));

    MPI_Bcast(&width,1,MPI_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&ws,1,MPI_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&ch,1,MPI_LONG,0,MPI_COMM_WORLD);
    int *blocks = (int*)malloc(world_size * sizeof(int));
    int block_size = (height + world_size - 1) / world_size;
    for ( int i = 0; i < world_size - 1; i++)
    { 
      blocks[i] = block_size;
    }
    blocks[world_size-1] = height - (world_size - 1)*block_size;
    MPI_Scatter(blocks,1,MPI_INT,&my_block_size,1,MPI_INT,0,MPI_COMM_WORLD);
    work = (unsigned char*)calloc((my_block_size+4)*ws,sizeof(unsigned char));
    int *disp = (int*)malloc(world_size * sizeof(int));
    if ( !disp || !work ) 
    {
      fprintf(stderr, "Error allocating memory\n");
      return -1;
    }
    for ( int i = 0; i < world_size; i++)
    { 
      blocks[i] = (blocks[i]+4) * ws;
      disp[i] = i * (block_size + 4) * ws;
    }
    MPI_Scatterv(data,blocks,disp,MPI_CHAR,work,blocks[0],MPI_CHAR,0,MPI_COMM_WORLD);
    free(disp);
    free(blocks);
  } else
  {
    MPI_Bcast(&width,1,MPI_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&ws,1,MPI_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&ch,1,MPI_LONG,0,MPI_COMM_WORLD);
    MPI_Scatter(NULL,1,MPI_INT,&my_block_size,1,MPI_INT,0,MPI_COMM_WORLD);
    work = (unsigned char*)calloc((my_block_size+4)*ws,sizeof(unsigned char));
    out = (unsigned char*)calloc(my_block_size*ws,sizeof(unsigned char));
    MPI_Scatterv(NULL,NULL,NULL,MPI_CHAR,work,(my_block_size+4)*ws,MPI_CHAR,0,MPI_COMM_WORLD);
  }


  // This is not portable to other MPI libraries
  char *comm_local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = atoi(comm_local_rank);
  char *comm_local_size = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  int local_size = atoi(comm_local_size);
  int num_devices = acc_get_num_devices(acc_device_nvidia);
#pragma acc set device_num(local_rank%num_devices) device_type(acc_device_nvidia)
  printf("DEBUG[%d]: using device %d\n", world_rank, acc_get_device_num(acc_device_nvidia));
    fflush(stdout);

  // Pre-allocate device and queues for timing
  //blur5_update(data,out,width,height, ch, ws,world_rank,world_size);
  blur5_pipelined(work,out,width,my_block_size, ch, ws);
  //blur5_pipelined_multi(data,out,width,height, ch, ws);
  bzero(out,my_block_size*ws);

  double st, et;

#if 0
  MPI_Barrier(MPI_COMM_WORLD);
  st = omp_get_wtime();
  blur5(data,out,width,height, ch, ws,world_rank,world_size);
  MPI_Barrier(MPI_COMM_WORLD);
  et = omp_get_wtime();
  //bzero(out,sz);
  if (world_rank==0)
    printf("Time (original): %lf seconds\n", (et-st));

  st = omp_get_wtime();
  blur5_blocked(data,out,width,height, ch, ws);
  et = omp_get_wtime();
  bzero(out,sz);
  printf("Time (blocked): %lf seconds\n", (et-st));

  MPI_Barrier(MPI_COMM_WORLD);
  st = omp_get_wtime();
  blur5_update(data,out,width,height, ch, ws, world_rank, world_size);
  et = omp_get_wtime();
  //bzero(out,sz);
  if (world_rank==0)
    printf("Time (update): %lf seconds\n", (et-st));
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  st = omp_get_wtime();
  blur5_pipelined(work,out,width,my_block_size, ch, ws);
  MPI_Barrier(MPI_COMM_WORLD);
  et = omp_get_wtime();
  if (world_rank==0)
    printf("Time (pipelined): %lf seconds\n", (et-st));

#if 0
  st = omp_get_wtime();
  blur5_pipelined_multi(data,out,width,height, ch, ws);
  et = omp_get_wtime();
  printf("Time (multi): %lf seconds\n", (et-st));
#endif

  // TODO: Add All_gatherv
  if (world_rank == 0 )
  {
    int *disp = (int*)malloc(world_size * sizeof(int));
    int *cnts = (int*)malloc(world_size * sizeof(int));
    if(!disp || !cnts) 
    {
      return -1;
    }
    int block_size = (height + world_size - 1) / world_size;
    for ( int i = 0; i < world_size-1; i++)
    { 
      cnts[i] = block_size * ws;
      disp[i] = i * block_size * ws;
    }
    cnts[world_size - 1] = (height - ((world_size-1)*block_size)) * ws;
    disp[world_size - 1] = (world_size-1) * block_size * ws;
    fflush(stdout);
    MPI_Gatherv(out,cnts[0],MPI_CHAR,out,cnts,disp,MPI_CHAR,0,MPI_COMM_WORLD);
    memcpy(img->imageData,out,width*height*ch);
    if(!cvSaveImage(argv[2],img))
      fprintf(stderr,"Failed to write to %s.\n",argv[2]);

    cvReleaseImage(&img);
    free(data);
  } else
  {
    MPI_Gatherv(out,my_block_size*ws,MPI_CHAR,0,0,0,MPI_CHAR,0,MPI_COMM_WORLD);
  }
  free(work);
  free(out);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
