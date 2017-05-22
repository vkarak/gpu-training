! First example, version 00

!****************************************************************

MODULE first_example_data

   IMPLICIT NONE

   INTEGER, PARAMETER :: N = 19

END MODULE first_example_data

!****************************************************************

PROGRAM first_example_v02

   USE first_example_data

   IMPLICIT NONE

   INTEGER :: a(N,N,N),b(N,N,N)

   INTEGER :: i,j,k
   INTEGER(kind=SELECTED_INT_KIND(18)) :: checksum,expected

!$acc data create(a,b)

!!$ Initialise the array
!$acc parallel loop
   DO k = 1,N
    DO j = 1,N
     DO i = 1,N
      a(i,j,k) = i*j*k
     ENDDO
    ENDDO
   ENDDO
!$acc end parallel loop

!!$ Process the array
   CALL double_array(a,b)

!!$ Calculate a checksum
   checksum = 0
!$acc parallel loop reduction(+:checksum)
   DO k = 1,N
    DO j = 1,N
     DO i = 1,N
      checksum = checksum + b(i,j,k)
     ENDDO
    ENDDO
   ENDDO
!$acc end parallel loop

!$acc end data

!!$ The expected total is 2*(N*(N+1)/2)**3
   expected = 2*(N*(N+1)/2)**3

!!$ Print the values obtained
   PRINT *,"Checksum value: ",checksum
   PRINT *,"Expected total: ",expected

!!$ Check if the result is correct
   IF (checksum == expected) THEN
    PRINT *,"Result: Correct!"
   ELSE
    PRINT *,"Result: Wrong."
   ENDIF

END PROGRAM first_example_v02

!****************************************************************
! This routine doubles a passed array of integers
!****************************************************************

SUBROUTINE double_array(c,d)

   USE first_example_data, ONLY:N

   IMPLICIT NONE

   INTEGER, INTENT(in) :: c(N,N,N)
   INTEGER, INTENT(out) :: d(N,N,N)

   INTEGER :: i,j,k

   INTEGER, EXTERNAL :: double_single

!$acc parallel loop present(c,d)
   DO k = 1,N
    DO j = 1,N
     DO i = 1,N
      d(i,j,k) = double_single(c(i,j,k))
     ENDDO
    ENDDO
   ENDDO
!$acc end parallel loop

END SUBROUTINE double_array

!****************************************************************
! This routine doubles a single, passed integer
!****************************************************************

FUNCTION double_single(e) RESULT(f)

   IMPLICIT NONE

   INTEGER, INTENT(in) :: e
   INTEGER :: f

   f = 2*e

END FUNCTION double_single

!****************************************************************
! EOF
