! First example, version 00

!****************************************************************

MODULE first_example_data

   IMPLICIT NONE

   INTEGER, PARAMETER :: N = 19

END MODULE first_example_data

!****************************************************************

PROGRAM first_example_v00

   USE first_example_data

   IMPLICIT NONE

   INTEGER :: a(N,N,N),b(N,N,N)

   INTEGER :: i,j,k
   INTEGER(kind=SELECTED_INT_KIND(18)) :: checksum,expected

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
!$acc parallel loop
   DO k = 1,N
    DO j = 1,N
     DO i = 1,N
      b(i,j,k) = 2*a(i,j,k)
     ENDDO
    ENDDO
   ENDDO
!$acc end parallel loop

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

END PROGRAM first_example_v00

!****************************************************************
! EOF
