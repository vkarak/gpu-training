module type_defs
  implicit none

  type level1
     integer i
     real r
     real, pointer,dimension(:) :: field1
     type (level2),pointer :: lvl2_ptr
  end type level1

  type level2
     logical flag
     real, allocatable,dimension(:) :: field2
  end type level2
end module type_defs
!============================================
!============================================
module openacc_transfers
! this module implements the routine for a simple manual deepcopy
  use type_defs
  use openacc
  implicit none

! these generic interfaces allow using the same name in calls in the main code, no matter which type
  interface acc_manual_copyin
     module procedure copyin_lvl1
     module procedure copyin_lvl2
  end interface acc_manual_copyin

  interface acc_manual_delete
     module procedure delete_lvl1
     module procedure delete_lvl2
  end interface acc_manual_delete

  interface acc_manual_copyout
     module procedure copyout_lvl1
     module procedure copyout_lvl2
  end interface acc_manual_copyout
contains

!============================================
! copyin routines
  subroutine copyin_lvl1(lvl1_var)
    type(level1):: lvl1_var

!$acc enter data copyin(lvl1_var)
!$acc enter data copyin(lvl1_var%field1)

    call acc_manual_copyin(lvl1_var%lvl2_ptr)
    call acc_attach(lvl1_var%lvl2_ptr)  ! this is necessary because inside the above call, no information is present anymore that the pointer argument is itself member of a parent type !
  end subroutine copyin_lvl1


  subroutine copyin_lvl2(lvl2_var)
    type(level2):: lvl2_var

!$acc enter data copyin(lvl2_var)
!$acc enter data copyin(lvl2_var%field2)
  end subroutine copyin_lvl2
!============================================
!delete routines
  subroutine delete_lvl1(lvl1_var)
    type(level1):: lvl1_var

!$acc exit data delete(lvl1_var%field1)

    call acc_manual_delete(lvl1_var%lvl2_ptr)
    call acc_detach(lvl1_var%lvl2_ptr) ! this is necessary because inside the above call, no information is present anymore that the pointer argument is itself member of a parent type !
!$acc exit data delete(lvl1_var)
  end subroutine delete_lvl1


  subroutine delete_lvl2(lvl2_var)
    type(level2):: lvl2_var

!$acc exit data delete(lvl2_var%field2)
!$acc exit data delete(lvl2_var)
  end subroutine delete_lvl2
!============================================
!copyout routines
  subroutine copyout_lvl1(lvl1_var)
    type(level1):: lvl1_var

!$acc exit data copyout(lvl1_var%field1)

    call acc_manual_copyout(lvl1_var%lvl2_ptr)
    call acc_detach(lvl1_var%lvl2_ptr) ! this is necessary because inside the above call, no information is present anymore that the pointer argument is itself member of a parent type !
!$acc exit data copyout(lvl1_var)
  end subroutine copyout_lvl1


  subroutine copyout_lvl2(lvl2_var)
    type(level2):: lvl2_var

!$acc exit data copyout(lvl2_var%field2)
!$acc exit data copyout(lvl2_var)
  end subroutine copyout_lvl2

end module openacc_transfers
!============================================
!============================================
program example
  use type_defs
  use openacc_transfers
  implicit none

  type(level1) :: var_lvl1
  type(level2),target :: var_lvl2
  type(level2):: another_lvl2

  integer,parameter:: nlen=20
  integer i

! allocate data structures with some predefined size
  allocate(var_lvl1%field1(nlen))
  allocate(var_lvl2%field2(nlen))
  allocate(another_lvl2%field2(nlen))
! let the pointer inside var_lvl1 point to var_lvl2
  var_lvl1%lvl2_ptr => var_lvl2

! initialize with some data
  var_lvl1%field1(:)=    1.0
  var_lvl2%field2(:)=    2.0
  another_lvl2%field2(:)=0.0

! begin manual deepcopy data region, using our own routines
  call acc_manual_copyin(var_lvl1)
  call acc_manual_copyin(another_lvl2)

!$acc parallel loop present(var_lvl1,another_lvl2) default(none)
  do i=1,nlen
     another_lvl2%field2(i) = var_lvl1%field1(i) + var_lvl1%lvl2_ptr%field2(i)
  enddo
!$acc end parallel

! end manual deepcopy data region, using our own routines
  call acc_manual_delete(var_lvl1)
  call acc_manual_copyout(another_lvl2)

! check if the field2 member is not zero anymore
  print*,'RESULT:'
  print*,another_lvl2%field2(:)

  deallocate(var_lvl1%field1)
  deallocate(var_lvl2%field2)
  deallocate(another_lvl2%field2)

end program example
