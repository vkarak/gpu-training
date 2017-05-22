// First example, C with dynamic allocation, version 01

/* Dynamically allocated, multidimensional arrays are hard to
 * handle in C using OpenACC v1.0.
 * The multidimensional array indices a[k][j][i] are handled 
 * as a chain of pointers. This makes it hard when copying the 
 * "array" to the accelerator as we need a "deep copy" that 
 * moves the entire pointer chain.

 * Upcoming versions of OpenACC will make this easier. In the 
 * meantime, we can use the Cray extended OpenACC runtime API
 * to manually carry out the deep copy.

 * This example shows you how. It is based on Example 2 of the 
 * Cray "openacc.examples" man page.

 * The use of the extended runtime API to create and free the
 * GPU memory replaces the use of a data region. There is no
 * version 00 of this example, as everything has to be done
 * explicitly and there is no automatic moving of data.

 * As the runtime API is only accessible when OpenACC directives 
 * are being recognised, we use the _OPENACC preprocessing macro 
 * to ensure the code still functions correctly on the host.
 */
 
//***************************************************************

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

  // We will need the runtime API
#include <openacc.h>
  /* A convenience macro to set a pointer on the accelerator
   * see man openacc.examples (Example 2) for details
   */
#define SET_ACC_PTR(acc_ptr, acc_target) ( \
   cray_acc_memcpy_to_device( &(acc_ptr), &(acc_target), sizeof(void*) ))

int main(){

  int i,j,k;
  int checksum,expected;

  // The problem size, not a parameter now.
  int N=19;

  //****************************************************************
  // The data arrays

  /* This is the USUAL (but not the best) way to dynamically allocate a
   * multidimensional array in C.  With this, we can only guarantee that
   * i-slices of the array are contiguous in memory. There is no guarantee
   * that, for instance, a[k][j][N-1] is contiguous with a[k][j+1][0].  This
   * may lead to bad cache usage on either CPU or accelerator.
   */

// The first data array
  int ***a = (int ***)malloc(N*sizeof(int **));
  for (k=0; k<N; k++) {
    a[k] = (int **)malloc(N*sizeof(int *));
    for (j=0; j<N; j++) {
      a[k][j] = (int *)malloc(N*sizeof(int));
    }
  }

// The second data array
  int ***b = (int ***)malloc(N*sizeof(int **));
  for (k=0; k<N; k++) {
    b[k] = (int **)malloc(N*sizeof(int *));
    for (j=0; j<N; j++) {
      b[k][j] = (int *)malloc(N*sizeof(int));
    }
  }

#ifdef _OPENACC
  //****************************************************************
  /* Now create the same array structures on the GPU using the 
   * Cray extended OpenACC runtime
   * see man openacc.examples (Example 2) for details.
   */
  /* create a[0:N] */
  int ***acc_a = (int ***)cray_acc_create(a,N*sizeof(int **));
  for (k=0; k<N; k++) {
    /* create a[k][0:N] */
    int **acc_ak = (int **)cray_acc_create(a[k],N*sizeof(int *));
    /* fix acc pointer acc_a[k][j] */
    SET_ACC_PTR(acc_a[k], acc_ak);
    for (j=0; j<N; j++) {
      int *acc_akj = (int *)cray_acc_create(a[k][j],N*sizeof(int));
      SET_ACC_PTR(acc_ak[j],acc_akj);
    }
  }

  /* Now do the same for array b */
  int ***acc_b = (int ***)cray_acc_create(b,N*sizeof(int **));
  for (k=0; k<N; k++) {
    int **acc_bk = (int **)cray_acc_create(b[k],N*sizeof(int *));
    SET_ACC_PTR(acc_b[k], acc_bk);
    for (j=0; j<N; j++) {
      int *acc_bkj = (int *)cray_acc_create(b[k][j],N*sizeof(int));
      SET_ACC_PTR(acc_bk[j],acc_bkj);
    }
  }
#endif /* _OPENACC */

  //****************************************************************

  // Initialise the array
  // Have to specify the sizes of the array to transfer here
#pragma acc parallel loop present(a[0:N])
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	a[k][j][i] = (i+1)*(j+1)*(k+1);

  // Process the array
#pragma acc parallel loop present(a[0:N],b[0:N])
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	b[k][j][i] = 2*a[k][j][i];

  // Calculate a checksum
  checksum = 0;
#pragma acc parallel loop reduction(+:checksum) present(b[0:N])
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	checksum += b[k][j][i];

#ifdef _OPENACC
  //****************************************************************
  // Perform a deep free of the accelerator memory structures
  for (k=0; k<N; k++) {
    for (j=0; j<N; j++) {
      acc_delete((void *)a[k][j], N);
      acc_delete((void *)b[k][j], N);
      //cscs cray_acc_delete((void *)a[k][j]);
      //cray_acc_delete((void *)b[k][j]);
    }
    acc_delete((void *) a[k], N);
    acc_delete((void *) b[k], N);
    //cray_acc_delete((void *) a[k]);
    //cray_acc_delete((void *) b[k]);
  }
  acc_delete((void *) a, N);
  acc_delete((void *) b, N);
  //cray_acc_delete((void *) a);
  //cray_acc_delete((void *) b);
#endif /* _OPENACC */

  // Free the CPU memory
  free(a);
  free(b);

  // The expected total is 2*(N*(N+1)/2)**3
  expected = 2*pow(N*(N+1)/2,3);

  // Print the values obtained
  printf("Checksum value: %d\n",checksum);
  printf("Expected total: %d\n",expected);

  // Check if the result is correct
  if(checksum == expected)
    printf("Result: Correct!\n");
  else
    printf("Result: Wrong.\n");

  return 0;
}

//****************************************************************
// EOF
