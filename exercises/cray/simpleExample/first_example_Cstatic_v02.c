// First example, C with static allocation, version 02

//****************************************************************

#include <stdio.h>
#include <math.h>

// The array extent in each dimension
#define N 19

void double_array(int c[N][N][N], int d[N][N][N]);
int double_single(int e);

//****************************************************************

void main(){

  // The data arrays
  int a[N][N][N],b[N][N][N];

  int i,j,k;
  int checksum,expected;

#pragma acc data create(a,b)
  {
  // Initialise the array
#pragma acc parallel loop
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	a[k][j][i] = (i+1)*(j+1)*(k+1);

  // Process the array
  double_array(a,b);

  // Calculate a checksum
  checksum = 0;
#pragma acc parallel loop reduction(+:checksum)
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	checksum += b[k][j][i];

  }
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
// This routine doubles a passed array of integers
//****************************************************************

void double_array(int c[N][N][N], int d[N][N][N]) {

  int i,j,k;

#pragma acc parallel loop present(c[0:N*N*N],d[0:N*N*N])
  for(k=0; k<N; k++)
    for(j=0; j<N; j++)
      for(i=0; i<N; i++)
	d[k][j][i] = double_single(c[k][j][i]);

}

//****************************************************************
// This routine doubles a single, passed integer
//****************************************************************

int double_single(int e){
  
  return 2*e;

}

//****************************************************************
// EOF
