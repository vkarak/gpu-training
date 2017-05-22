!!$*********************************************************************

!!$ This benchmark test program is measuring a cpu performance
!!$ of floating point operation by a Poisson equation solver.

!!$ If you have any question, please ask me via email.
!!$ written by Ryutaro HIMENO, November 26, 2001.
!!$ Version 3.0
!!$ ----------------------------------------------
!!$ Ryutaro Himeno, Dr. of Eng.
!!$ Head of Computer Information Division,
!!$ RIKEN (The Institute of Pysical and Chemical Research)
!!$ Email : himeno@postman.riken.go.jp
!!$ ---------------------------------------------------------------
!!$ You can adjust the size of this benchmark code to fit your target
!!$ computer. In that case, please chose following sets of
!!$ (mimax,mjmax,mkmax):
!!$ small : 65,33,33
!!$ small : 129,65,65
!!$ midium: 257,129,129
!!$ large : 513,257,257
!!$ ext.large: 1025,513,513
!!$ This program is to measure a computer performance in MFLOPS
!!$ by using a kernel which appears in a linear solver of pressure
!!$ Poisson eq. which appears in an incompressible Navier-Stokes solver.
!!$ A point-Jacobi method is employed in this solver as this method can 
!!$ be easyly vectrized and be parallelized.
!!$ ------------------
!!$ Finite-difference method, curvilinear coodinate system
!!$ Vectorizable and parallelizable on each grid point
!!$ No. of grid points : imax x jmax x kmax including boundaries
!!$ ------------------
!!$ A,B,C:coefficient matrix, wrk1: source term of Poisson equation
!!$ wrk2 : working area, OMEGA : relaxation parameter
!!$ BND:control variable for boundaries and objects ( = 0 or 1)
!!$ P: pressure
!!$ -------------------

MODULE himeno_include

   IMPLICIT NONE

!!$ Some precision labels
   INTEGER, PARAMETER :: dp = KIND(1d0)
   INTEGER, PARAMETER :: sp = KIND(1e0)

#ifdef DOUBLE_PRECISION
   INTEGER, PARAMETER :: fp = dp
#else
   INTEGER, PARAMETER :: fp = sp
#endif

!!$ Problem sizes
#if PROBLEM_SIZE == 4
   INTEGER, PARAMETER :: mimax=513,mjmax=257,mkmax=257
#endif
#if PROBLEM_SIZE == 3
   INTEGER, PARAMETER :: mimax=257,mjmax=129,mkmax=129
#endif
#if PROBLEM_SIZE == 2
   INTEGER, PARAMETER :: mimax=129,mjmax=65,mkmax=65
#endif
#if PROBLEM_SIZE == 1
   INTEGER, PARAMETER :: mimax=65,mjmax=33,mkmax=33
#endif

!!$ Arrays
   REAL(kind=fp) :: p(mimax,mjmax,mkmax)
   REAL(kind=fp) :: a(mimax,mjmax,mkmax,4), &
        b(mimax,mjmax,mkmax,3), &
        c(mimax,mjmax,mkmax,3)
   REAL(kind=fp) :: bnd(mimax,mjmax,mkmax)
   REAL(kind=fp) :: wrk1(mimax,mjmax,mkmax), &
        wrk2(mimax,mjmax,mkmax)
!!$ Other constants
   INTEGER :: imax,jmax,kmax
   REAL(kind=fp) :: omega

CONTAINS

!!$*************************************************************
   FUNCTION gettime() RESULT(rtime)
!!$*************************************************************

      IMPLICIT NONE

      INTEGER(kind=SELECTED_INT_KIND(18)) :: ic,ir,im
      REAL(kind=dp) :: rtime

      CALL SYSTEM_CLOCK(ic,ir,im)

      rtime = REAL(ic,kind=dp)/REAL(ir,kind=dp)

   END FUNCTION gettime
!!$*************************************************************

END MODULE himeno_include

PROGRAM main

   USE himeno_include

#ifdef _OPENMP
   USE omp_lib
#endif

   IMPLICIT NONE

   REAL(kind=dp) :: gosa
   REAL(kind=dp) :: cpu0, cpu1, flop, xmflops2, score 

   REAL(kind=dp) :: ttarget = 6d0

   INTEGER :: nn


   omega = 0.8d0
   imax = mimax-1
   jmax = mjmax-1
   kmax = mkmax-1

!!$ Initializing matrixes
   CALL initmt

   PRINT *
#ifdef DOUBLE_PRECISION
   PRINT *,"Double precision"
#else
   PRINT *,"Single precision"
#endif

   PRINT *
   PRINT *,"PROBLEM_SIZE : ",PROBLEM_SIZE
   PRINT *,'  mimax = ',mimax,' mjmax = ',mjmax,' mkmax = ',mkmax
   PRINT *,'  imax = ',imax,' jmax = ',jmax,' kmax = ',kmax

#ifdef _OPENACC
   PRINT *
   PRINT *,"Using OpenACC"
   PRINT *,"  NTPB : ",NTPB
#endif

#ifdef _OPENMP
   PRINT *
   PRINT *,"Using OpenMP"
#endif

!$omp target data &
!$omp& map(to:a,b,c,bnd,wrk1)&
!$omp& map(to:p) map(alloc:wrk2) 

!!$ Start measuring
   nn = 3
!!$ Jacobi iteration
   cpu0 = gettime()
   CALL jacobi(nn,gosa)
   cpu1 = gettime()
   cpu1 = cpu1 - cpu0

!!$ Calculate performance
   flop = REAL(kmax-2)*REAL(jmax-2)*REAL(imax-2)*34d0*REAL(nn)
   xmflops2 = flop/cpu1 * 1.0d-6

   PRINT *
   PRINT *,"Iterations  : ",nn
   PRINT *,"Time (secs) : ",cpu1
   PRINT '(" Gosa       : ",E24.18," ")',gosa
   PRINT *,"MFLOPS      : ",xmflops2

!!$ Now estimate how many iterations could be done in ttarget seconds
   nn = INT(ttarget/(cpu1/nn))
!!$ Hardwire for consistency when testing
   nn = 100

!!$ Jacobi iteration
   cpu0 = gettime()
   CALL jacobi(nn,gosa)
   cpu1 = gettime()
   cpu1 = cpu1 - cpu0
!$omp end target data
   
!!$ Calculate performance
   flop = REAL(kmax-2)*REAL(jmax-2)*REAL(imax-2)*34d0*REAL(nn)
   xmflops2 = flop/cpu1 * 1.0d-6
   score = xmflops2/82.84d0

   PRINT *
   PRINT *,"Iterations  : ",nn
   PRINT *,"Time (secs) : ",cpu1
   PRINT '(" Gosa       : ",E24.18," ")',gosa
   PRINT *,"MFLOPS      : ",xmflops2
   PRINT *,'Score based on Pentium III 600MHz :',score



END PROGRAM main

!!$**************************************************************
SUBROUTINE initmt
!!$**************************************************************
      
   USE himeno_include

   IMPLICIT NONE

   INTEGER :: i,j,k

   DO k = 1,mkmax
    DO j = 1,mjmax
     DO i = 1,mimax
      a(i,j,k,1) = 0d0
      a(i,j,k,2) = 0d0
      a(i,j,k,3) = 0d0
      a(i,j,k,4) = 0d0
      b(i,j,k,1) = 0d0
      b(i,j,k,2) = 0d0
      b(i,j,k,3) = 0d0
      c(i,j,k,1) = 0d0
      c(i,j,k,2) = 0d0
      c(i,j,k,3) = 0d0
      p(i,j,k)   = 0d0
      wrk1(i,j,k)= 0d0   
      bnd(i,j,k) = 0d0 
     ENDDO
    ENDDO
   ENDDO

   DO k = 1,kmax
    DO j = 1,jmax
     DO i = 1,imax
      a(i,j,k,1) = 1d0
      a(i,j,k,2) = 1d0
      a(i,j,k,3) = 1d0
      a(i,j,k,4) = 1d0/6d0
      b(i,j,k,1) = 0d0
      b(i,j,k,2) = 0d0
      b(i,j,k,3) = 0d0
      c(i,j,k,1) = 1d0
      c(i,j,k,2) = 1d0
      c(i,j,k,3) = 1d0
      p(i,j,k) = ((k-1)*(k-1))/((kmax-1)*(kmax-1)*1d0)
      wrk1(i,j,k) = 0d0
      bnd(i,j,k) = 1d0
     ENDDO
    ENDDO
   ENDDO

END SUBROUTINE initmt

!!$*************************************************************
SUBROUTINE jacobi(nn,gosa)
!!$*************************************************************

   USE himeno_include

   IMPLICIT NONE

   INTEGER, INTENT(in) :: nn
   REAL(kind=dp), INTENT(out) :: gosa

   INTEGER :: loop,i,j,k
   REAL(kind=fp) :: s0,ss
   REAL(kind=dp) :: gosa1


!$omp target data map(to:a,b,c,bnd,wrk1) &
!$omp& map(tofrom:p) map(alloc:wrk2)
   iter_loop: DO loop = 1,nn
    gosa = 0d0

    gosa1 = 0d0


!$omp target teams distribute reduction(+:gosa1) 
    DO k = 2,kmax-1
     DO j = 2,jmax-1
      DO i = 2,imax-1
       S0 = a(i,j,k,1) *   p(i+1,j  ,k  ) &
          + a(i,j,k,2) *   p(i  ,j+1,k  ) &
          + a(i,j,k,3) *   p(i  ,  j,k+1) &
          + b(i,j,k,1) * ( p(i+1,j+1,k  ) - p(i+1,j-1,k  ) &
                         - p(i-1,j+1,k  ) + p(i-1,j-1,k  ) ) &
          + b(i,j,k,2) * ( p(i  ,j+1,k+1) - p(i  ,j-1,k+1) &
                         - p(i  ,j+1,k-1) + p(i  ,j-1,k-1)) &
          + b(i,j,k,3) * ( p(i+1,j  ,k+1) - p(i-1,j  ,k+1) &
                         - p(i+1,j  ,k-1) + p(i-1,j  ,k-1)) &
          + c(i,j,k,1) *   p(i-1,j  ,k  ) &
          + c(i,j,k,2) *   p(i  ,j-1,k  ) &
          + c(i,j,k,3) *   p(i  ,j  ,k-1) &
          + wrk1(i,j,k)

       ss = ( s0 * a(i,j,k,4) - p(i,j,k) ) * bnd(i,j,k)
       gosa1 = gosa1 + ss*ss
       wrk2(i,j,k) = p(i,j,k) + omega * ss
      ENDDO
     ENDDO
    ENDDO
!$omp end target teams distribute

!$omp target teams distribute
    DO k = 2,kmax-1
     DO j = 2,jmax-1
      DO i = 2,imax-1
       p(i,j,k) = wrk2(i,j,k)
      ENDDO
     ENDDO
    ENDDO
!$omp end target teams distribute

    gosa = gosa + gosa1
   ENDDO iter_loop
!$omp end target data

END SUBROUTINE jacobi
