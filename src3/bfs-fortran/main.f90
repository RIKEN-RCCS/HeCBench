program bfs
  use omp_lib
  implicit none
  integer :: nodes,edges,source,i,id,cost,unit,ios
  integer,allocatable :: start(:),degree(:),graph(:),edge_cost(:),device_cost(:),host_cost(:)
  integer(kind=1),allocatable :: mask(:),updating(:),visited(:)
  character(len=1024)::file
  if(command_argument_count()/=1) then; print *,'Usage: ./main <input_file>'; stop 1; end if
  call get_command_argument(1,file)
  open(newunit=unit,file=trim(file),status='old',action='read',iostat=ios)
  if(ios/=0) then; print *,'Error Reading graph file ',trim(file); stop 1; end if
  read(unit,*) nodes; allocate(start(0:nodes-1),degree(0:nodes-1),mask(0:nodes-1),updating(0:nodes-1),visited(0:nodes-1))
  do i=0,nodes-1; read(unit,*)start(i),degree(i); end do
  read(unit,*)source; source=0; read(unit,*)edges; allocate(graph(0:edges-1),edge_cost(0:edges-1),device_cost(0:nodes-1),host_cost(0:nodes-1))
  read(unit,*,iostat=ios) (graph(i),edge_cost(i),i=0,edges-1)
  if (ios /= 0) then
    print *,'Error Reading graph edges'
    stop 1
  end if
  close(unit)
  mask=0; updating=0; visited=0; mask(source)=1; visited(source)=1; device_cost=-1; device_cost(source)=0
  print '(a,i0,a)','run bfs (#nodes = ',nodes,') on device'
  call bfs_gpu(nodes,start,degree,edges,graph,mask,updating,visited,device_cost)
  mask=0; updating=0; visited=0; mask(source)=1; visited(source)=1; host_cost=-1; host_cost(source)=0
  print '(a,i0,a)','run bfs (#nodes = ',nodes,') on host (cpu)'
  call bfs_cpu(nodes,start,degree,graph,mask,updating,visited,host_cost)
  if(all(device_cost==host_cost)) then; print *,'PASS'; else; print *,'FAIL'; end if
  deallocate(edge_cost)
contains
  subroutine bfs_gpu(n,start,degree,e,graph,mask,updating,visited,cost)
    integer,intent(in)::n,e,start(0:),degree(0:),graph(0:); integer,intent(inout)::cost(0:)
    integer(kind=1),intent(inout)::mask(0:),updating(0:),visited(0:)
    integer(kind=1)::over(0:0); integer::tid,k,id; real(8)::t0,t1,total
    total=0.d0
    !$omp target data map(to:start(0:n-1),degree(0:n-1),graph(0:e-1),visited(0:n-1),mask(0:n-1),updating(0:n-1)) map(alloc:over(0:0)) map(tofrom:cost(0:n-1))
    do
      over(0)=0; !$omp target update to(over(0:0)); t0=omp_get_wtime()
      !$omp target teams distribute parallel do thread_limit(256) private(k,id)
      do tid=0,n-1
        if(mask(tid)/=0) then
          mask(tid)=0
          do k=start(tid),start(tid)+degree(tid)-1
            id=graph(k)
            if(visited(id)==0) then; cost(id)=cost(tid)+1; updating(id)=1; end if
          end do
        end if
      end do
      !$omp end target teams distribute parallel do
      !$omp target teams distribute parallel do thread_limit(256)
      do tid=0,n-1
        if(updating(tid)/=0) then; mask(tid)=1; visited(tid)=1; over(0)=1; updating(tid)=0; end if
      end do
      !$omp end target teams distribute parallel do
      t1=omp_get_wtime(); total=total+(t1-t0); !$omp target update from(over(0:0))
      if(over(0)==0) exit
    end do
    print '(a,f12.6,a)','Total kernel execution time : ',total*1.d6,' (us)'
    !$omp end target data
  end subroutine
  subroutine bfs_cpu(n,start,degree,graph,mask,updating,visited,cost)
    integer,intent(in)::n,start(0:),degree(0:),graph(0:); integer,intent(inout)::cost(0:)
    integer(kind=1),intent(inout)::mask(0:),updating(0:),visited(0:)
    integer(kind=1)::over; integer::tid,k,id
    do; over=0
      do tid=0,n-1
        if(mask(tid)/=0) then; mask(tid)=0; do k=start(tid),start(tid)+degree(tid)-1; id=graph(k); if(visited(id)==0) then; cost(id)=cost(tid)+1; updating(id)=1; end if; end do; end if
      end do
      do tid=0,n-1; if(updating(tid)/=0) then; mask(tid)=1; visited(tid)=1; over=1; updating(tid)=0; end if; end do
      if(over==0) exit
    end do
  end subroutine
end program
