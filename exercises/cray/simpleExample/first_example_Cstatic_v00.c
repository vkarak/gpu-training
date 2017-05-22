// First example, C with static allocation, version 00

//****************************************************************

#include <stdio.h>
#include <math.h>

// The array extent in each dimension
#define N 19

void main(){

  // The data arrays
  int a[N][N][N],b[N][N][N];

  int i,j,k;
  int checksum,expected;

  // Initialise the array
#pragma acc parallel loop
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	a[k][j][i] = (i+1)*(j+1)*(k+1);

  // Process the array
#pragma acc parallel loop
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	b[k][j][i] = 2*a[k][j][i];

  // Calculate a checksum
  checksum = 0;
#pragma acc parallel loop reduction(+:checksum)
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	checksum += b[k][j][i];

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

}

//****************************************************************
// EOF
