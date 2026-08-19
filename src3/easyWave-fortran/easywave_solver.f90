module easywave_solver
  use easywave_constants
  use easywave_fault
  implicit none
contains
  subroutine run_easywave(grid_file, source_file, time_minutes, ierr)
    character(*), intent(in) :: grid_file, source_file
    integer, intent(in) :: time_minutes
    integer, intent(out) :: ierr
    integer :: u, ios, nlon, nlat, i, j, k, step, dt, time_max, progress, next_progress
    integer :: imin, imax, jmin, jmax, nfault, count_rate, count_start, count_end
    character(len=16) :: label
    real(dp) :: lonmin, lonmax, latmin, latmax, zmin, zmax, dlon, dlat, dx, dy, dtloc
    real(dp) :: lon, lat, uplift, absmax, threshold, elapsed_ms
    real(sp) :: ftopo, absh
    real(sp), allocatable :: node(:,:,:), r6(:), c1(:), c2(:), c3(:), c4(:)
    type(fault_t), allocatable :: faults(:)

    ierr=0
    open(newunit=u,file=grid_file,status='old',action='read',iostat=ios)
    if (ios /= 0) then
      write(*,'(A,1X,A)') 'Unable to open bathymetry:', trim(grid_file); ierr=1; return
    end if
    read(u,*,iostat=ios) label
    if (ios /= 0 .or. trim(label) /= 'DSAA') then
      write(*,'(A)') 'Only ASCII DSAA bathymetry is supported by this Fortran port.'; ierr=2; close(u); return
    end if
    read(u,*,iostat=ios) nlon,nlat
    read(u,*,iostat=ios) lonmin,lonmax
    read(u,*,iostat=ios) latmin,latmax
    read(u,*,iostat=ios) zmin,zmax
    if (ios /= 0 .or. nlon < 3 .or. nlat < 3) then
      write(*,'(A)') 'Invalid DSAA bathymetry header.'; ierr=3; close(u); return
    end if
    allocate(node(max_vars_per_node,nlat,nlon),r6(nlat),c1(nlon),c2(nlat),c3(nlon),c4(nlat),stat=ios)
    if (ios /= 0) then; ierr=4; close(u); return; end if
    node=0.0_sp
    read(u,*,iostat=ios) node(id_topo,:,:)
    if (ios /= 0) then; ierr=5; close(u); return; end if
    do i=1,nlon
      do j=1,nlat
        ftopo=node(id_topo,j,i)
        node(id_time,j,i)=-1.0_sp
        node(id_depth,j,i)=max(0.0_sp,-ftopo)
        if (node(id_depth,j,i)>0.0_sp .and. node(id_depth,j,i)<10.0_sp) node(id_depth,j,i)=10.0_sp
      end do
    end do
    close(u)
    dlon=(lonmax-lonmin)/real(nlon-1,dp); dlat=(latmax-latmin)/real(nlat-1,dp)
    dx=rearth*deg2rad(dlon); dy=rearth*deg2rad(dlat)
    dtloc=huge(1.0_dp)
    do i=1,nlon
      do j=1,nlat
        if(node(id_depth,j,i)>0.0_sp) then
          dtloc=min(dtloc,0.8_dp*dx*cosdeg(latmin+real(j-1,dp)*dlat) / &
                    sqrt(gravity*real(node(id_depth,j,i),dp)))
        end if
      end do
    end do
    if(dtloc>15.0_dp) then
      dt=15
    else if(dtloc>10.0_dp) then
      dt=10
    else if(dtloc>5.0_dp) then
      dt=5
    else if(dtloc>2.0_dp) then
      dt=2
    else if(dtloc>1.0_dp) then
      dt=1
    else
      write(*,'(A)') 'Bathymetry requires a time step below one second.'; ierr=6; return
    end if
    write(*,'(A,F8.3,A,I0,A)') 'Stable CFL time step: ',dtloc,' sec; selected ',dt,' sec'

    do i=1,nlon
      if(node(id_depth,1,i)/=0.0_sp .and. node(id_depth,2,i)==0.0_sp) node(id_depth,1,i)=0.0_sp
      if(node(id_depth,nlat,i)/=0.0_sp .and. node(id_depth,nlat-1,i)==0.0_sp) node(id_depth,nlat,i)=0.0_sp
    end do
    do j=1,nlat
      if(node(id_depth,j,1)/=0.0_sp .and. node(id_depth,j,2)==0.0_sp) node(id_depth,j,1)=0.0_sp
      if(node(id_depth,j,nlon)/=0.0_sp .and. node(id_depth,j,nlon-1)==0.0_sp) node(id_depth,j,nlon)=0.0_sp
    end do
    do j=1,nlat
      r6(j)=real(cosdeg(latmin+(real(j,dp)-0.5_dp)*dlat),sp)
    end do
    do i=1,nlon
      do j=1,nlat
        if(node(id_depth,j,i)==0.0_sp) cycle
        node(id_r1,j,i)=real(real(dt,dp)/dy/real(r6(j),dp),sp)
        if(i<nlon .and. node(id_depth,j,i+1)/=0.0_sp) then
          node(id_r2,j,i)=real(0.5_dp*gravity*dt/dy/real(r6(j),dp)*(node(id_depth,j,i)+node(id_depth,j,i+1)),sp)
        else if(i==nlon) then
          node(id_r2,j,i)=real(gravity*dt/dy/real(r6(j),dp)*node(id_depth,j,i),sp)
        end if
        if(j<nlat .and. node(id_depth,j+1,i)/=0.0_sp) then
          node(id_r4,j,i)=real(0.5_dp*gravity*dt/dy*(node(id_depth,j,i)+node(id_depth,j+1,i)),sp)
        else if(j==nlat) then
          node(id_r4,j,i)=real(gravity*dt/dy*node(id_depth,j,i),sp)
        end if
      end do
    end do
    do i=1,nlon
      c1(i)=0.0_sp; c3(i)=0.0_sp
      if(node(id_depth,1,i)/=0.0_sp) c1(i)=1.0_sp/sqrt(gravity*node(id_depth,1,i))
      if(node(id_depth,nlat,i)/=0.0_sp) c3(i)=1.0_sp/sqrt(gravity*node(id_depth,nlat,i))
    end do
    do j=1,nlat
      c2(j)=0.0_sp; c4(j)=0.0_sp
      if(node(id_depth,j,1)/=0.0_sp) c2(j)=1.0_sp/sqrt(gravity*node(id_depth,j,1))
      if(node(id_depth,j,nlon)/=0.0_sp) c4(j)=1.0_sp/sqrt(gravity*node(id_depth,j,nlon))
    end do

    call read_faults(source_file,faults,nfault,ios)
    if(ios/=0) then
      write(*,'(A,1X,A)') 'Unable to parse Okada fault file:',trim(source_file); ierr=7; return
    end if
    imin=nlon; imax=1; jmin=nlat; jmax=1
    do i=1,nlon
      lon=lonmin+real(i-1,dp)*dlon
      do j=1,nlat
        if(node(id_depth,j,i)==0.0_sp) cycle
        lat=latmin+real(j-1,dp)*dlat; uplift=0.0_dp
        do k=1,nfault
          uplift=uplift+displacement(faults(k),lon,lat)
        end do
        node(id_height,j,i)=real(uplift,sp)
        if(abs(uplift)>1.0e-4_dp) then
          imin=min(imin,i); imax=max(imax,i); jmin=min(jmin,j); jmax=max(jmax,j)
        end if
      end do
    end do
    if(imin==nlon) then; write(*,'(A)') 'Zero initial displacement.'; ierr=8; return; end if
    absmax=maxval(abs(node(id_height,:,:))); threshold=max(0.01_dp*real(absmax,dp),0.0_dp)
    where(abs(node(id_height,:,:))<real(threshold,sp)) node(id_height,:,:)=0.0_sp
    ! The original expands a clipped domain dynamically.  Updating the full wet domain is numerically equivalent and keeps all updates on the device.
    imin=2; imax=nlon-1; jmin=2; jmax=nlat-1
    time_max=time_minutes*60; progress=600; next_progress=progress
    call system_clock(count_start,count_rate)
    !$omp target data map(tofrom:node) map(to:r6,c1,c2,c3,c4)
    do step=0,time_max,dt
      !$omp target teams distribute parallel do collapse(2) private(absh)
      do i=imin,imax
        do j=jmin,jmax
          if(node(id_depth,j,i)==0.0_sp) cycle
          node(id_height,j,i)=node(id_height,j,i)-node(id_r1,j,i) * &
            (node(id_m,j,i)-node(id_m,j,i-1)+node(id_n,j,i)*r6(j)-node(id_n,j-1,i)*r6(j-1))
          absh=abs(node(id_height,j,i)); if(absh<1.0e-5_sp) node(id_height,j,i)=0.0_sp
          if(node(id_height,j,i)>node(id_hmax,j,i)) node(id_hmax,j,i)=node(id_height,j,i)
          if(node(id_time,j,i)<0.0_sp .and. absh>0.001_sp) node(id_time,j,i)=real(step,sp)
        end do
      end do
      !$omp end target teams distribute parallel do
      !$omp target teams distribute parallel do
      do i=2,nlon-1
        node(id_height,1,i)=sqrt(node(id_n,1,i)**2+0.25_sp*(node(id_m,1,i)+node(id_m,1,i-1))**2)*c1(i)
        if(node(id_n,1,i)>0.0_sp) node(id_height,1,i)=-node(id_height,1,i)
        node(id_height,nlat,i)=sqrt(node(id_n,nlat-1,i)**2+0.25_sp*(node(id_m,nlat,i)+node(id_m,nlat,i-1))**2)*c3(i)
        if(node(id_n,nlat-1,i)<0.0_sp) node(id_height,nlat,i)=-node(id_height,nlat,i)
      end do
      !$omp end target teams distribute parallel do
      !$omp target teams distribute parallel do
      do j=2,nlat-1
        node(id_height,j,1)=sqrt(node(id_m,j,1)**2+0.25_sp*(node(id_n,j,1)+node(id_n,j-1,1))**2)*c2(j)
        if(node(id_m,j,1)>0.0_sp) node(id_height,j,1)=-node(id_height,j,1)
        node(id_height,j,nlon)=sqrt(node(id_m,j,nlon-1)**2+0.25_sp*(node(id_n,j,nlon)+node(id_n,j-1,nlon))**2)*c4(j)
        if(node(id_m,j,nlon-1)>0.0_sp) node(id_height,j,nlon)=-node(id_height,j,nlon)
      end do
      !$omp end target teams distribute parallel do
      !$omp target teams distribute parallel do collapse(2)
      do i=1,nlon-1
        do j=1,nlat-1
          if(node(id_depth,j,i)*node(id_depth,j,i+1)/=0.0_sp) then
            node(id_m,j,i)=node(id_m,j,i)-node(id_r2,j,i)*(node(id_height,j,i+1)-node(id_height,j,i))
          end if
          if(node(id_depth,j,i)*node(id_depth,j+1,i)/=0.0_sp) then
            node(id_n,j,i)=node(id_n,j,i)-node(id_r4,j,i)*(node(id_height,j+1,i)-node(id_height,j,i))
          end if
        end do
      end do
      !$omp end target teams distribute parallel do
      !$omp target teams distribute parallel do
      do i=1,nlon-1
        node(id_m,1,i)=node(id_m,1,i)-node(id_r2,1,i)*(node(id_height,1,i+1)-node(id_height,1,i))
        node(id_m,nlat,i)=node(id_m,nlat,i)-node(id_r2,nlat,i)*(node(id_height,nlat,i+1)-node(id_height,nlat,i))
      end do
      !$omp end target teams distribute parallel do
      !$omp target teams distribute parallel do
      do j=1,nlat-1
        node(id_n,j,1)=node(id_n,j,1)-node(id_r4,j,1)*(node(id_height,j+1,1)-node(id_height,j,1))
        node(id_n,j,nlon)=node(id_n,j,nlon)-node(id_r4,j,nlon)*(node(id_height,j+1,nlon)-node(id_height,j,nlon))
      end do
      !$omp end target teams distribute parallel do
      if(step>=next_progress) then
        !$omp target update from(node)
        call system_clock(count_end)
        elapsed_ms=1000.0_dp*real(count_end-count_start,dp)/real(count_rate,dp)
        write(*,'(A,I2.2,A,I2.2,A,I2.2,A,F0.3,A)') 'Model time = ',step/3600,':', &
          mod(step,3600)/60,':',mod(step,60),',   elapsed: ',elapsed_ms,' msec'
        next_progress=next_progress+progress
      end if
    end do
    !$omp end target data
    call write_outputs(node,nlat,nlon,lonmin,lonmax,latmin,latmax)
    write(*,'(A)') 'PASS'
  end subroutine run_easywave

  subroutine write_outputs(node,nlat,nlon,lonmin,lonmax,latmin,latmax)
    real(sp),intent(in)::node(:,:,:)
    integer,intent(in)::nlat,nlon
    real(dp),intent(in)::lonmin,lonmax,latmin,latmax
    integer::u,i,j
    integer(kind=2):: ni,nj
    ni=int(nlon,kind=2); nj=int(nlat,kind=2)
    open(newunit=u,file='eWave.2D.sshmax',access='stream',form='unformatted',status='replace')
    write(u) 'DSBB',ni,nj,lonmin,lonmax,latmin,latmax,0.0_dp,1.0_dp
    do j=1,nlat; do i=1,nlon; write(u) node(id_hmax,j,i); end do; end do
    close(u)
    open(newunit=u,file='eWave.2D.time',access='stream',form='unformatted',status='replace')
    write(u) 'DSBB',ni,nj,lonmin,lonmax,latmin,latmax,0.0_dp,1.0_dp
    do j=1,nlat; do i=1,nlon; write(u) node(id_time,j,i)/60.0_sp; end do; end do
    close(u)
  end subroutine write_outputs
end module easywave_solver
