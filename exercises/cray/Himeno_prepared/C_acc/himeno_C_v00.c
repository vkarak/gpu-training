/*********************************************************************
      This benchmark test program is measuring a cpu performance 
      of floating point operation and memory access speed.  

      Modification needed for testing turget computer!!
      Please adjust parameter : nn to take one minute to execute
      all calculation.  Original parameter set is for PC with 
      200 MHz MMX PENTIUM, whose score using this benchmark test
      is about 32.3 MFLOPS.

      If you have any question, please ask me via email.
      written by Ryutaro HIMENO, October 3, 1998.
      Version 2.0  
      ----------------------------------------------
         Ryutaro Himeno, Dr. of Eng.
         Head of Computer Information Center, 
         The Institute of Pysical and Chemical Research (RIKEN)
         Email : himeno@postman.riken.go.jp
      ---------------------------------------------------------------
      You can adjust the size of this benchmark code to fit your target
      computer.  In that case, please chose following sets of 
      (mimax,mjmax,mkmax):
       small : 129,65,65
       midium: 257,129,129
       large : 513,257,257
       ext.large: 1025,513,513
      This program is to measure a computer performance in MFLOPS
      by using a kernel which appears in a linear solver of pressure 
      Poisson included in an incompressible Navier-Stokes solver. 
      A point-Jacobi method is employed in this solver.
      ------------------
      Finite-difference method, curvilinear coodinate system
      Vectorizable and parallelizable on each grid point
      No. of grid points : imax x jmax x kmax including boundaries
      ------------------
      A,B,C:coefficient matrix, wrk1: source term of Poisson equation
      wrk2 : working area, OMEGA : relaxation parameter
      BND:control variable for boundaries and objects ( = 0 or 1)
      P: pressure 
       -----------------
      -------------------     
      "use portlib" statement on the next line is for Visual fortran 
      to use UNIX libraries.  Please remove it if your system is UNIX.
      -------------------
     use portlib

     Version 0.2 
*********************************************************************/

#include <stdio.h>

#ifdef DOUBLE_PRECISION
typedef double real;
#else
typedef float real;
#endif

#if PROBLEM_SIZE == 4
#define MIMAX            513
#define MJMAX            257
#define MKMAX            257
#endif

#if PROBLEM_SIZE == 3
#define MIMAX            257
#define MJMAX            129
#define MKMAX            129
#endif

#if PROBLEM_SIZE == 2
#define MIMAX            129
#define MJMAX            65
#define MKMAX            65
#endif

#if PROBLEM_SIZE == 1
#define MIMAX            65
#define MJMAX            33
#define MKMAX            33
#endif

static real  p[MKMAX][MJMAX][MIMAX];
static real  a[4][MKMAX][MJMAX][MIMAX],
             b[3][MKMAX][MJMAX][MIMAX],
             c[3][MKMAX][MJMAX][MIMAX];
static real  bnd[MKMAX][MJMAX][MIMAX];
static real  wrk1[MKMAX][MJMAX][MIMAX],
             wrk2[MKMAX][MJMAX][MIMAX];

static int imax, jmax, kmax;
static real omega;

double gettime();
double jacobi(int);
void initmt();

int main()
{
#include <omp.h>

  int i, j, k;
  // The reduction variable should be double precision always
  double gosa;
  double cpu0, cpu1, flop, xmflops2, score;

  double ttarget=6.0;

  int nthreads;



  omega = 0.8;
  imax = MIMAX-1;
  jmax = MJMAX-1;
  kmax = MKMAX-1;

  // Initializing matrixes
  initmt();

#ifdef DOUBLE_PRECISION
  printf("\n Double precision\n");
#else
  printf("\n Single precision\n");
#endif

  printf("\n PROBLEM_SIZE : %d\n",PROBLEM_SIZE);
  printf("   mimax = %d mjmax = %d mkmax = %d\n",MIMAX, MJMAX, MKMAX);
  printf("   imax = %d jmax = %d kmax = %d\n",imax,jmax,kmax);

#ifdef _OPENACC
  printf("\n Using OpenACC\n");
  printf("   NTPB : %d\n",NTPB);
#endif

#ifdef _OPENMP
  printf("\n Using OpenMP\n");
#pragma omp parallel
  nthreads = omp_get_num_threads();
  printf("   OMP_NUM_THREADS : %d\n",nthreads);
#endif

  // Start measuring
  int nn = 3;

  // Jacobi iteration
  cpu0 = gettime();
  gosa = jacobi(nn);
  cpu1 = gettime();
  cpu1 = cpu1 - cpu0;

  // Calculate performance
  flop = (double)(kmax-2)*(double)(jmax-2)*(double)(imax-2)*34.0*(double)nn;
  xmflops2 = (flop/cpu1) * 1.0e-6;

  printf("\n Iterations  : %d\n",nn);
  printf(  " Time (secs) : %f sec.\n", cpu1);
  printf(  " Gosa        : %22.17e \n",gosa);
  printf(  " MFLOPS      : %f\n",xmflops2);
  
  // Now estimate how many iterations could be done in ttarget seconds
  nn = (int)ttarget/(cpu1/(real)nn);
  // Hardwire for consistency when testing
  nn = 100;

  // Jacobi iteration
  cpu0 = gettime();
  gosa = jacobi(nn);
  cpu1 = gettime();
  cpu1 = cpu1 - cpu0;

  // Calculate performance
  flop = (double)(kmax-2)*(double)(jmax-2)*(double)(imax-2)*34.0*(double)nn;
  xmflops2 = (flop/cpu1) * 1.0e-6;
  score = xmflops2/82.84;
  
  printf("\n Iterations  : %d\n",nn);
  printf(  " Time (secs) : %f sec.\n", cpu1);
  printf(  " Gosa        : %22.17e \n",gosa);
  printf(  " MFLOPS      : %f\n",xmflops2);
  printf(" Score based on Pentium III 600MHz : %f\n",score);
  


  return (0);
}

void initmt()
{
	int i,j,k;


#pragma omp parallel default(shared) private(i,j,k)
	{

#pragma omp for
  for(k=0 ; k<MKMAX ; ++k)
    for(j=0 ; j<MJMAX ; ++j)
      for(i=0 ; i<MIMAX ; ++i){
        a[0][k][j][i]=0.0;
        a[1][k][j][i]=0.0;
        a[2][k][j][i]=0.0;
        a[3][k][j][i]=0.0;
        b[0][k][j][i]=0.0;
        b[1][k][j][i]=0.0;
        b[2][k][j][i]=0.0;
        c[0][k][j][i]=0.0;
        c[1][k][j][i]=0.0;
        c[2][k][j][i]=0.0;
        p[k][j][i]=0.0;
        wrk1[k][j][i]=0.0;
        bnd[k][j][i]=0.0;
      }


#pragma omp for
  for(k=0 ; k<kmax ; ++k)
    for(j=0 ; j<jmax ; ++j)
      for(i=0 ; i<imax ; ++i){
        a[0][k][j][i]=1.0;
        a[1][k][j][i]=1.0;
        a[2][k][j][i]=1.0;
        a[3][k][j][i]=1.0/6.0;
        b[0][k][j][i]=0.0;
        b[1][k][j][i]=0.0;
        b[2][k][j][i]=0.0;
        c[0][k][j][i]=1.0;
        c[1][k][j][i]=1.0;
        c[2][k][j][i]=1.0;
        p[k][j][i]=(real)( (k*k)/(double)((kmax-1)*(kmax-1)) );
        wrk1[k][j][i]=0.0;
        bnd[k][j][i]=1.0;
      }

	} /* end omp parallel/acc data region */
}

double jacobi(int nn)
{
  int i,j,k,loop;
  real s0;
  // Accumulation variables should be double precision
  double gosa, gosa1, ss;



#pragma omp parallel default(shared) private(loop,k,j,i,s0,ss,gosa1)
  {
  for(loop=0;loop<nn;++loop){
#pragma omp barrier
#pragma omp master
    gosa = 0.0;

    gosa1 = 0.0;


#pragma omp for
    for(k=1 ; k<kmax-1 ; ++k)
      for(j=1 ; j<jmax-1 ; ++j)
        for(i=1 ; i<imax-1 ; ++i){
          s0 = a[0][k][j][i] *   p[k  ][j  ][i+1]
             + a[1][k][j][i] *   p[k  ][j+1][i  ]
             + a[2][k][j][i] *   p[k+1][j  ][i  ]
             + b[0][k][j][i] * ( p[k  ][j+1][i+1] - p[k  ][j-1][i+1]
                               - p[k  ][j+1][i-1] + p[k  ][j-1][i-1] )
             + b[1][k][j][i] * ( p[k+1][j+1][i  ] - p[k+1][j-1][i  ]
                               - p[k-1][j+1][i  ] + p[k-1][j-1][i  ] )
             + b[2][k][j][i] * ( p[k+1][j  ][i+1] - p[k+1][j  ][i-1]
                               - p[k-1][j  ][i+1] + p[k-1][j  ][i-1] )
             + c[0][k][j][i] *   p[k  ][j  ][i-1]
             + c[1][k][j][i] *   p[k  ][j-1][i  ]
             + c[2][k][j][i] *   p[k-1][j  ][i  ]
             + wrk1[k][j][i];

          ss = ( s0 * a[3][k][j][i] - p[k][j][i] ) * bnd[k][j][i];

          gosa1 = gosa1 + ss*ss;

          wrk2[k][j][i] = p[k][j][i] + omega * ss;
        }


#pragma omp for nowait
    for(k=1 ; k<kmax-1 ; ++k)
      for(j=1 ; j<jmax-1 ; ++j)
        for(i=1 ; i<imax-1 ; ++i)
          p[k][j][i] = wrk2[k][j][i];

#pragma omp critical
    gosa = gosa + gosa1;

  } /* end iteration loop */

  } /* end omp parallel/acc data region */

  return(gosa);
}

double gettime()
{
#include <sys/time.h>

  struct timeval tm;
  double t ;

  static int base_sec = 0,base_usec = 0;

  gettimeofday(&tm, NULL);
  
  if(base_sec == 0 && base_usec == 0)
    {
      base_sec = tm.tv_sec;
      base_usec = tm.tv_usec;
      t = 0.0;
  } else {
    t = (double) (tm.tv_sec-base_sec) + 
      ((double) (tm.tv_usec-base_usec))/1.0e6 ;
  }

  return t ;
}
