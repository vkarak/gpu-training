!*********************************************************************

! This benchmark test program is measuring a cpu performance
! of floating point operation by a Poisson equation solver.

! If you have any question, please ask me via email.
! written by Ryutaro HIMENO, November 26, 2001.
! Version 3.0
! ----------------------------------------------
! Ryutaro Himeno, Dr. of Eng.
! Head of Computer Information Division,
! RIKEN (The Institute of Pysical and Chemical Research)
! Email : himeno@postman.riken.go.jp
! -----------------------------------------------------------
! You can adjust the size of this benchmark code to fit your target
! computer. In that case, please chose following sets of
! (mimax,mjmax,mkmax):
! small : 65,33,33
! small : 129,65,65
! midium: 257,129,129
! large : 513,257,257
! ext.large: 1025,513,513
! This program is to measure a computer performance in MFLOPS
! by using a kernel which appears in a linear solver of pressure
! Poisson eq. which appears in an incompressible Navier-Stokes solver.
! A point-Jacobi method is employed in this solver as this method can 
! be easyly vectrized and be parallelized.
! ------------------
! Finite-difference method, curvilinear coodinate system
! Vectorizable and parallelizable on each grid point
! No. of grid points : imax x jmax x kmax including boundaries
! ------------------
! A,B,C:coefficient matrix, wrk1: source term of Poisson equation
! wrk2 : working area, OMEGA : relaxation parameter
! BND:control variable for boundaries and objects ( = 0 or 1)
! P: pressure
! -------------------
PROGRAM HIMENOBMTXP

   IMPLICIT NONE

   INCLUDE 'mpif.h'
   INCLUDE 'param.h'

!     ttarget specifys the measuring period in sec
   REAL (kind=kind_sp), PARAMETER :: ttarget=60.0

   INTEGER :: mx,my,mz,it,nn,ierr
   REAL (kind=kind_dp) :: gosa,score

   REAL (kind=kind_dp) :: cpu,cpu0,cpu1,xmflops2,flop

   omega = REAL(1,kind=kind_rl)/REAL(8,kind=kind_rl)
   mx = mx0-1
   my = my0-1
   mz = mz0-1

! Initializing communicator
   CALL initcomm

! Initializaing computational index
   CALL initmax(mx,my,mz,it)

!$omp target data map(alloc:p,a,b,c,wrk1,wrk2,bnd) &
!$omp      map(alloc:xsnd_up,xsnd_dn,xrcv_up,xrcv_dn) &
!$omp      map(alloc:ysnd_up,ysnd_dn,yrcv_up,yrcv_dn) &
!$omp      map(alloc:zsnd_up,zsnd_dn,zrcv_up,zrcv_dn)

! Initializing matrixes
   CALL initmt(mz,it)

! Write information from rank 0
   rubric: IF (id .EQ. 0) THEN

    PRINT *,"Using PROBLEM_SIZE =",PROBLEM_SIZE
    PRINT *

    WRITE(*,*) 'Sequential version array size'
    WRITE(*,*) ' mimax=',mx0,' mjmax=',my0,' mkmax=',mz0
    WRITE(*,*) 'Parallel version  array size'
    WRITE(*,*) ' mimax=',mimax,' mjmax=',mjmax,' mkmax=',mkmax
    WRITE(*,*) ' imax=',imax,' jmax=',jmax,' kmax=',kmax
    WRITE(*,*) ' I-decomp= ',ndx,' J-decomp= ',ndy, &
         ' K-decomp= ',ndz
    WRITE(*,*)

#ifdef DOUBLE_PRECISION
    PRINT *,"Using precision: double"
#else
    PRINT *,"Using precision: single"
#endif
    PRINT *

    PRINT *,"MPI buffersizes:"
    PRINT '("  x-dirn:",f8.2," kB")',jmax*kmax*float_size/1024d0
    PRINT '("  y-dirn:",f8.2," kB")',imax*kmax*float_size/1024d0
    PRINT '("  z-dirn:",f8.2," kB")',imax*jmax*float_size/1024d0
    PRINT *

#ifdef WAITANY
    PRINT *,"Using MPI_WAITANY"
#else
    PRINT *,"Using MPI_WAITALL"
#endif
    PRINT *

#ifdef _OPENMP
    PRINT *,"Compiling with OpenMP support: _OPENMP =",_OPENMP
#else
    PRINT *,"NOT compiling with OpenMP support"
#endif
    PRINT *

   ENDIF rubric
      
! Start measuring; calibrate with 

   nn = 3
   IF (id .EQ. 0) THEN
    WRITE(*,*) ' Start rehearsal measurement process.'
    WRITE(*,*) ' Measure the performance for ',nn,' iterations.'
   END IF

   gosa = 0
   cpu = 0
   CALL mpi_barrier(mpi_comm_world,ierr)
   cpu0 = mpi_wtime()
! Jacobi iteration
   CALL jacobi(nn,gosa)
   cpu1 = mpi_wtime() - cpu0

   CALL mpi_allreduce(cpu1,cpu,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,ierr)

   flop = REAL(mx-2)*REAL(my-2)*REAL(mz-2)*34d0
   IF (cpu .NE. 0.0) xmflops2=flop/cpu*1.0d-6*REAL(nn)
   IF (id .EQ. 0) THEN
!    WRITE(*,*) '  Gosa :',gosa
!    WRITE(*,*) '  MFLOPS:',xmflops2,'  time(s):',cpu
    PRINT *
    PRINT *,"  Calibration: Iterations  : ",nn
    PRINT *,"  Calibration: Time (secs) : ",cpu
    PRINT '("   Calibration:  Gosa       : ",E24.18," ")',gosa
    PRINT *,"  Calibration: MFLOPS      : ",xmflops2
    PRINT *
   ENDIF

   nn = INT(ttarget/(cpu/3.0))
! CRAY: hardwire the number of iterations, so we should
!       always get the same answer for gosa
   nn = 200

!     end the test loop
   IF (id .EQ. 0) THEN
    WRITE(*,*) 'Now, start the actual measurement process.'
    WRITE(*,*) 'The loop will be excuted in',nn,' times.'
    WRITE(*,*) 'This will take about one minute.'
    WRITE(*,*) 'Wait for a while.'
   ENDIF

   gosa = 0
   cpu = 0
   CALL mpi_barrier(mpi_comm_world,ierr)
   cpu0 = mpi_wtime()
! Jacobi iteration
   CALL jacobi(nn,gosa)

   cpu1= mpi_wtime() - cpu0

!$omp end target data

   CALL mpi_reduce(cpu1,cpu,1,MPI_REAL8,MPI_MAX,0,MPI_COMM_WORLD,ierr)

   IF (id .EQ. 0) THEN
    IF (cpu .NE. 0.0)  xmflops2=flop*1.0d-6/cpu*REAL(nn)

!    score = xmflops2/82.84
    PRINT *
    PRINT *,"  Measurement: Iterations  : ",nn
    PRINT *,"  Measurement: Time (secs) : ",cpu
    PRINT '("   Measurement:  Gosa       : ",E24.18," ")',gosa
    PRINT *,"  Measurement: MFLOPS      : ",xmflops2
!    PRINT *,'  Measurement: Score based on Pentium III 600MHz :',score
    PRINT *
   ENDIF

   CALL mpi_finalize(ierr)

END PROGRAM HIMENOBMTXP

!**************************************************************

!**************************************************************
SUBROUTINE initmt(mz,it)

   IMPLICIT NONE

   INCLUDE 'mpif.h'
   INCLUDE 'param.h'

   INTEGER :: mz,it
   INTEGER :: i,j,k

!$omp target teams distribute
   DO k=1,mkmax
    DO j=1,mjmax
     DO i=1,mimax
      a(i,j,k,1)=0
      a(i,j,k,2)=0
      a(i,j,k,3)=0
      a(i,j,k,4)=0
      b(i,j,k,1)=0
      b(i,j,k,2)=0
      b(i,j,k,3)=0
      c(i,j,k,1)=0
      c(i,j,k,2)=0
      c(i,j,k,3)=0
      p(i,j,k)=0
      wrk1(i,j,k)=0   
      wrk2(i,j,k)=0   
      bnd(i,j,k)=0 
     ENDDO
    ENDDO
   ENDDO
!$omp end target teams distribute

!$omp target teams distribute 
   DO k=1,kmax
    DO j=1,jmax
     DO i=1,imax
      a(i,j,k,1)=1
      a(i,j,k,2)=1
      a(i,j,k,3)=1
      a(i,j,k,4)=REAL(1,kind=kind_rl)/REAL(6,kind=kind_rl)
      b(i,j,k,1)=0
      b(i,j,k,2)=0
      b(i,j,k,3)=0
      c(i,j,k,1)=1
      c(i,j,k,2)=1
      c(i,j,k,3)=1
      p(i,j,k)=REAL((k-1+it)*(k-1+it),kind=kind_rl) &
           /REAL((mz-1)*(mz-1),kind=kind_rl)
      wrk1(i,j,k)=0   
      wrk2(i,j,k)=0  
      bnd(i,j,k)=1
     ENDDO
    ENDDO
   ENDDO
!$omp end target teams distribute

END SUBROUTINE initmt

!*************************************************************

!*************************************************************
SUBROUTINE jacobi(nn,gosa)

   IMPLICIT NONE

   INCLUDE 'mpif.h'
   INCLUDE 'param.h'

   INTEGER :: nn,i,j,k,loop,ierr
   REAL (kind=kind_dp) :: gosa,wgosa
   REAL (kind=kind_rl) :: s0,ss

   INTEGER :: ireq_coll

! initialize receive buffers
! The edge processes in the processor grid
! will not have their receive buffers packed, so
! we should make sure that the receive buffers here are 
! pre-packed with the correct data. 
! I may as well do this on all the ranks; on the internal
! ranks the data will just get over-written.
! If I am not doing G2G, it is the host version that
! should be correct

   IF (ndx .GT. 1) THEN
!$omp target teams distribute
    DO k=1,kmax
     DO j=1,jmax
      xrcv_dn(j,k)=p(1,j,k)
      xrcv_up(j,k)=p(imax,j,k)
     ENDDO
    ENDDO
!$omp end target teams distribute
#ifndef G2G_recv
!$omp target update from(xrcv_dn)
!$omp target update from(xrcv_up)
#endif
   ENDIF !ndx

   IF (ndy .GT. 1) THEN
!$omp target teams distribute
    DO k=1,kmax
     DO i=1,imax
      yrcv_dn(i,k)=p(i,1,k)
      yrcv_up(i,k)=p(i,jmax,k)
     ENDDO
    ENDDO
!$omp end target teams distribute
#ifndef G2G_recv
!$omp target update from(yrcv_dn)
!$omp target update from(yrcv_up)
#endif
   ENDIF !ndy

   IF (ndz .GT. 1) THEN
!$omp target teams distribute
    DO j=1,jmax
     DO i=1,imax
      zrcv_dn(i,j)=p(i,j,1)
      zrcv_up(i,j)=p(i,j,kmax)
     ENDDO
    ENDDO
!$omp end target teams distribute
#ifndef G2G_recv
!$omp target update from(zrcv_dn)
!$omp target update from(zrcv_up)
#endif
   ENDIF !ndz

   iter_lp: DO loop=1,nn

    gosa = 0
    wgosa = 0
!!$ Prepost the MPI receives
    CALL sendp_recv()

!$omp target teams distribute private (s0,ss) reduction(+:wgosa)
    DO K=2,kmax-1
     DO J=2,jmax-1
      DO I=2,imax-1
       S0 = a(I,J,K,1)*p(I+1,J,K)+a(I,J,K,2)*p(I,J+1,K) &
            +a(I,J,K,3)*p(I,J,K+1) &
            +b(I,J,K,1)*(p(I+1,J+1,K)-p(I+1,J-1,K) &
            -p(I-1,J+1,K)+p(I-1,J-1,K)) &
            +b(I,J,K,2)*(p(I,J+1,K+1)-p(I,J-1,K+1) &
            -p(I,J+1,K-1)+p(I,J-1,K-1)) &
            +b(I,J,K,3)*(p(I+1,J,K+1)-p(I-1,J,K+1) &
            -p(I+1,J,K-1)+p(I-1,J,K-1)) &
            +c(I,J,K,1)*p(I-1,J,K)+c(I,J,K,2)*p(I,J-1,K) &
            +c(I,J,K,3)*p(I,J,K-1)+wrk1(I,J,K)
       SS = (S0*a(I,J,K,4)-p(I,J,K))*bnd(I,J,K)
       WGOSA = WGOSA + SS*SS
       wrk2(I,J,K) = p(I,J,K) + OMEGA * SS
      ENDDO
     ENDDO
    ENDDO
!$omp end target teams distribute

!!$ These buffer packs could be done with one kernel for each
!!$ buffer, or one kernel for each direction. At the moment, 
!!$ best performance is from using the latter.
    IF (ndx .GT. 1) THEN
!$omp target teams distribute
     DO k=1,kmax
      DO j=1,jmax
       xsnd_dn(j,k)=wrk2(2,j,k)
       xsnd_up(j,k)=wrk2(imax-1,j,k)
      ENDDO
     ENDDO
!$omp end target teams distribute
#ifndef G2G_send
!$omp target update from(xsnd_dn,xsnd_up)
#endif
    ENDIF !ndx
       
    IF (ndy .GT. 1) THEN
!$omp target teams distribute
     DO k=1,kmax
      DO i=1,imax
       ysnd_dn(i,k)=wrk2(i,2,k)
       ysnd_up(i,k)=wrk2(i,jmax-1,k)
      ENDDO
     ENDDO
!$omp end target teams distribute
#ifndef G2G_send
!$omp target update from(ysnd_dn,ysnd_up)
#endif
    ENDIF !ndy

    IF (ndz .GT. 1) THEN
!$omp target teams distribute
     DO j=1,jmax
      DO i=1,imax
       zsnd_dn(i,j)=wrk2(i,j,2)
       zsnd_up(i,j)=wrk2(i,j,kmax-1)
      ENDDO
     ENDDO
!$omp end target teams distribute
#ifndef G2G_send
!$omp target update from(zsnd_dn,zsnd_up)
#endif
    ENDIF !ndz

!$omp target teams distribute
    DO K=2,kmax-1
     DO J=2,jmax-1
      DO I=2,imax-1
       p(I,J,K)=wrk2(I,J,K)
      ENDDO
     ENDDO
    ENDDO
!$omp end target teams distribute

!!$ Make sure each direction of buffers has been packed before
!!$ sending via MPI.
    IF (ndx .GT. 1) THEN
     CALL sendp1_send()
    ENDIF
    IF (ndy .GT. 1) THEN
     CALL sendp2_send()
    ENDIF
    IF (ndz .GT. 1) THEN
     CALL sendp3_send()
    ENDIF

    CALL mpi_iallreduce(wgosa,gosa,1,MPI_REAL8,MPI_SUM, &
         MPI_COMM_WORLD,ireq_coll,ierr)

!!$ Wait for completion of all halo exchange MPI
#ifdef WAITANY
    CALL sendp_waitany_update()
#else
    CALL sendp_waitall_update()
#endif

!      Unpack the receive buffers on the GPU in the background...
    IF (ndx .GT. 1) THEN
!$omp target teams distribute
     DO k=1,kmax
      DO j=1,jmax
       p(1,   j,k)=xrcv_dn(j,k)
       p(imax,j,k)=xrcv_up(j,k)
      ENDDO
     ENDDO
!$omp end target teams distribute
    ENDIF !ndx

    IF (ndy .GT. 1) THEN
!$omp target teams distribute
     DO k=1,kmax
      DO i=1,imax
       p(i,1,   k)=yrcv_dn(i,k)
       p(i,jmax,k)=yrcv_up(i,k)
      ENDDO
     ENDDO
!$omp end target teams distribute
    ENDIF !ndy

    IF (ndz .GT. 1) THEN
!$omp target teams distribute
     DO j=1,jmax
      DO i=1,imax
       p(i,j,1   )=zrcv_dn(i,j)
       p(i,j,kmax)=zrcv_up(i,j)
      ENDDO
     ENDDO
!$omp end target teams distribute
    ENDIF !ndz

!!$ Make sure the async collective has completed
    CALL MPI_WAIT(ireq_coll,ist,ierr)

!   IF (id .EQ. 0) PRINT *,"gosa",loop,gosa,wgosa

!!$ StreamX, StreamY, StreamZ will be continued in the next iteration.
   ENDDO iter_lp
!C End of iteration

END SUBROUTINE jacobi

!****************************************************************

!****************************************************************
SUBROUTINE initcomm

   IMPLICIT NONE
   
   INCLUDE 'mpif.h'
   INCLUDE 'param.h'
   
   LOGICAL :: ipd(3),ir
   INTEGER :: idm(3),ierr,icomm
   
   CALL mpi_init(ierr)
   CALL mpi_comm_size(mpi_comm_world,npe,ierr)
   CALL mpi_comm_rank(mpi_comm_world,id,ierr)
   
   IF (ndx*ndy*ndz .NE. npe) THEN
    IF (id .EQ. 0) THEN
     WRITE(*,*) 'Invalid number of PE'
     WRITE(*,*) 'Please check partitioning pattern'
     WRITE(*,*) '                 or number of  PE'
    END IF
    CALL mpi_finalize(ierr)
    STOP
   END IF
   
   icomm= mpi_comm_world

   idm(1:3) = [ndx,ndy,ndz]

   ipd(1:3) = .FALSE.

   ir = .FALSE.

   CALL mpi_cart_create(icomm,ndims,idm,ipd,ir,mpi_comm_cart,ierr)
   CALL mpi_cart_get(mpi_comm_cart,ndims,idm,ipd,iop,ierr)

   IF (ndz .GT. 1) &
        CALL mpi_cart_shift(mpi_comm_cart,2,1,npz(1),npz(2),ierr)

   IF (ndy .GT. 1) &
        CALL mpi_cart_shift(mpi_comm_cart,1,1,npy(1),npy(2),ierr)

   IF (ndx .GT. 1) &
        CALL mpi_cart_shift(mpi_comm_cart,0,1,npx(1),npx(2),ierr)

END SUBROUTINE initcomm

!****************************************************************

!****************************************************************
SUBROUTINE initmax(mx,my,mz,ks)
   
   IMPLICIT NONE

   INCLUDE 'mpif.h'
   INCLUDE 'param.h'
   
   INTEGER :: mx,my,mz
   INTEGER :: itmp,ks
   INTEGER :: mx1(0:ndx),my1(0:ndy),mz1(0:ndz)
   INTEGER :: mx2(0:ndx),my2(0:ndy),mz2(0:ndz)
   INTEGER :: i,j,k,ierr

!    define imax, communication direction
   itmp = mx/ndx
   mx1(0) = 0
   DO  i=1,ndx
    IF (i .LE. MOD(mx,ndx)) THEN
     mx1(i) = mx1(i-1) + itmp + 1
    ELSE
     mx1(i) = mx1(i-1) + itmp
    ENDIF
   ENDDO
   DO i=0,ndx-1
    mx2(i) = mx1(i+1) - mx1(i)
    IF (i .NE. 0)     mx2(i) = mx2(i) + 1
    IF (i .NE. ndx-1) mx2(i) = mx2(i) + 1
   ENDDO

   itmp = my/ndy
   my1(0) = 0
   DO  i=1,ndy
    IF (i .LE. MOD(my,ndy)) THEN
     my1(i) = my1(i-1) + itmp + 1
    ELSE
     my1(i) = my1(i-1) + itmp
    ENDIF
   ENDDO
   DO i=0,ndy-1
    my2(i) = my1(i+1) - my1(i)
    IF (i .NE. 0)      my2(i) = my2(i) + 1
    IF (i .NE. ndy-1)  my2(i) = my2(i) + 1
   ENDDO

   itmp = mz/ndz
   mz1(0) = 0
   DO  i=1,ndz
    IF (i .LE. MOD(mz,ndz)) THEN
     mz1(i) = mz1(i-1) + itmp + 1
    ELSE
     mz1(i) = mz1(i-1) + itmp
    ENDIF
   ENDDO
   DO i=0,ndz-1
    mz2(i) = mz1(i+1) - mz1(i)
    IF (i .NE. 0)      mz2(i) = mz2(i) + 1
    IF (i .NE. ndz-1)  mz2(i) = mz2(i) + 1
   ENDDO

   imax = mx2(iop(1))
   jmax = my2(iop(2))
   kmax = mz2(iop(3))

   IF (iop(3) .EQ. 0) THEN
    ks = mz1(iop(3))
   ELSE
    ks = mz1(iop(3)) - 1
   ENDIF

END SUBROUTINE initmax

!****************************************************************

!****************************************************************
SUBROUTINE sendp_recv()

   IMPLICIT NONE

   INCLUDE "mpif.h"
   INCLUDE "param.h"
   
   INTEGER :: i

!      Set the requests to null, in case some directions are not sending
   ireq_send(:) = MPI_REQUEST_NULL
   ireq_recv(:) = MPI_REQUEST_NULL

   CALL sendp3_recv()
   CALL sendp2_recv()
   CALL sendp1_recv()

END SUBROUTINE sendp_recv
!****************************************************************

!!$ Wait for all messages to complete and then transfer and unpack
!!$ halo buffers

!****************************************************************
SUBROUTINE sendp_waitall_update()

   IMPLICIT NONE
   
   INCLUDE "mpif.h"
   INCLUDE "param.h"
   
   INTEGER :: ierr
      
!      Wait on all the messages
   CALL mpi_waitall(6,ireq_recv,ist,ierr)

#ifndef G2G_recv
   IF (ndx > 1) THEN
!$omp target update to(xrcv_dn)
!$omp target update to(xrcv_up)
    CONTINUE
   ENDIF
   IF (ndy > 1) THEN
!$omp target update to(yrcv_dn)
!$omp target update to(yrcv_up)
    CONTINUE
   ENDIF
   IF (ndz > 1) THEN
!$omp target update to(zrcv_up)
!$omp target update to(zrcv_dn)
    CONTINUE
   ENDIF
#endif

   CALL mpi_waitall(6,ireq_send,ist,ierr)

END SUBROUTINE sendp_waitall_update
!****************************************************************
      
!!$ Use waitall if we are not using OpenMP device constructs, or we have
!!$ OpenMP device constructs with G2G receives (in both cases MPI delivers
!!$ the data where we need it).
!!$ If we have OpenMP device constructs without G2G receives, as soon as a 
!!$ message arrives, we start transferring it to the GPU.

!****************************************************************
SUBROUTINE sendp_waitany_update()

   IMPLICIT NONE
   
   INCLUDE "mpif.h"
   INCLUDE "param.h"

   INTEGER :: ierr,index,icount

!      Wait on all the messages
!!$ We have to copy data back to the GPU manually, as each message
!!$ completes, we should copy it back to the GPU.
!!$ CONTINUE statements needed to ensure code is compilable even
!!$ without OpenMP
   DO icount = 1,6
    CALL mpi_waitany(6,ireq_recv,index,ist,ierr)

    SELECT CASE(index)
    CASE(1)
     IF (ndx > 1) THEN
#ifndef G2G_recv
!$omp target update to(xrcv_up)
#endif
      CONTINUE
     ENDIF
    CASE(2)
     IF (ndx > 1) THEN
#ifndef G2G_recv
!$omp target update to(xrcv_dn)
#endif
      CONTINUE
     ENDIF
    CASE(3)
     IF (ndy > 1) THEN
#ifndef G2G_recv
!$omp target update to(yrcv_up)
#endif
      CONTINUE
     ENDIF
    CASE(4)
     IF (ndy > 1) THEN
#ifndef G2G_recv
!$omp target update to(yrcv_dn)
#endif
      CONTINUE
     ENDIF
    CASE(5)
     IF (ndz > 1) THEN
#ifndef G2G_recv
!$omp target update to(zrcv_up)
#endif
      CONTINUE
     ENDIF
    CASE(6)
     IF (ndz > 1) THEN
#ifndef G2G_recv
!$omp target update to(zrcv_dn)
#endif
      CONTINUE
     ENDIF
!       CASE(MPI_UNDEFINED)
!        exit
    END SELECT
   ENDDO

!!$ Now make sure the sends have also completed
   CALL mpi_waitall(6,ireq_send,ist,ierr)

END SUBROUTINE sendp_waitany_update
!****************************************************************
      
!****************************************************************
SUBROUTINE sendp3_recv()

   IMPLICIT NONE
      
   INCLUDE 'mpif.h'
   INCLUDE 'param.h'

   INTEGER :: ierr

   IF (ndz .EQ. 1) RETURN

#ifdef G2G_recv
!$omp target data use_device_ptr(zrcv_up,zrcv_dn)
#endif
   CALL mpi_irecv(zrcv_up,imax*jmax,mpi_realkind,npz(2),1+8, &
        mpi_comm_cart,ireq_recv(1+4),ierr)

   CALL mpi_irecv(zrcv_dn,imax*jmax,mpi_realkind,npz(1),2+8, &
        mpi_comm_cart,ireq_recv(2+4),ierr)
#ifdef G2G_recv
!$omp end target data
#endif

END SUBROUTINE sendp3_recv
!****************************************************************

!****************************************************************
SUBROUTINE sendp3_send()

   IMPLICIT NONE

   INCLUDE 'mpif.h'
   INCLUDE 'param.h'

   INTEGER :: ierr

   IF (ndz .EQ. 1) RETURN

#ifdef G2G_send
!$omp target data use_device_ptr(zsnd_dn,zsnd_up)
#endif
   CALL mpi_isend(zsnd_dn,imax*jmax,mpi_realkind,npz(1),1+8, &
        mpi_comm_cart,ireq_send(1+4),ierr)

   CALL mpi_isend(zsnd_up,imax*jmax,mpi_realkind,npz(2),2+8, &
        mpi_comm_cart,ireq_send(2+4),ierr)
#ifdef G2G_send
!$omp end target data
#endif

END SUBROUTINE sendp3_send
!****************************************************************

!****************************************************************
SUBROUTINE sendp2_recv()

   IMPLICIT NONE

   INCLUDE 'mpif.h'
   INCLUDE 'param.h'

   INTEGER :: ierr

   IF (ndy .EQ. 1) RETURN

#ifdef G2G_recv
!$omp target data use_device_ptr(yrcv_up,yrcv_dn)
#endif
   CALL mpi_irecv(yrcv_dn,imax*kmax,mpi_realkind,npy(1),2+4, &
        mpi_comm_cart,ireq_recv(2+2),ierr)

   CALL mpi_irecv(yrcv_up,imax*kmax,mpi_realkind,npy(2),1+4, &
        mpi_comm_cart,ireq_recv(1+2),ierr)
#ifdef G2G_recv
!$omp end target data
#endif

END SUBROUTINE sendp2_recv
!****************************************************************

!****************************************************************
SUBROUTINE sendp2_send()

   IMPLICIT NONE

   INCLUDE 'mpif.h'
   INCLUDE 'param.h'
      
   INTEGER :: ierr

   IF (ndy .EQ. 1) RETURN

#ifdef G2G_send
!$omp target data use_device_ptr(ysnd_dn,ysnd_up)
#endif
   CALL mpi_isend(ysnd_dn,imax*kmax,mpi_realkind,npy(1),1+4, &
        mpi_comm_cart,ireq_send(1+2),ierr)
   
   CALL mpi_isend(ysnd_up,imax*kmax,mpi_realkind,npy(2),2+4, &
        mpi_comm_cart,ireq_send(2+2),ierr)
#ifdef G2G_send
!$omp end target data
#endif

END SUBROUTINE sendp2_send
!****************************************************************

!****************************************************************
SUBROUTINE sendp1_recv()

   IMPLICIT NONE
   
   INCLUDE 'mpif.h'
   INCLUDE 'param.h'
   
   INTEGER :: ierr

   IF (ndx .EQ. 1) RETURN

#ifdef G2G_recv
!$omp target data use_device_ptr(xrcv_dn,xrcv_up)
#endif
   CALL mpi_irecv(xrcv_dn,jmax*kmax,mpi_realkind,npx(1),2, &
        mpi_comm_cart,ireq_recv(2+0),ierr)

   CALL mpi_irecv(xrcv_up,jmax*kmax,mpi_realkind,npx(2),1, &
        mpi_comm_cart,ireq_recv(1+0),ierr)
#ifdef G2G_recv
!$omp end target data
#endif

END SUBROUTINE sendp1_recv
!****************************************************************

!****************************************************************
SUBROUTINE sendp1_send()

   IMPLICIT NONE
   
   INCLUDE 'mpif.h'
   INCLUDE 'param.h'
   
   INTEGER :: ierr
   
   IF (ndx .EQ. 1) RETURN
   
#ifdef G2G_send
!$omp target data use_device_ptr(xsnd_dn,xsnd_up)
#endif
   CALL mpi_isend(xsnd_dn,jmax*kmax,mpi_realkind,npx(1),1, &
        mpi_comm_cart,ireq_send(1+0),ierr)
   
   CALL mpi_isend(xsnd_up,jmax*kmax,mpi_realkind,npx(2),2, &
        mpi_comm_cart,ireq_send(2+0),ierr)
#ifdef G2G_send
!$omp end target data
#endif

END SUBROUTINE sendp1_send

!****************************************************************
! EOF

