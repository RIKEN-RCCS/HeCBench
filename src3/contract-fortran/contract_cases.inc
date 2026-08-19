select case(case_id)
case(1)
 a=x;b=y;do d=0,nv-1;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do;end do
case(2)
 a=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;do b=0,nv-1;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end do;end if;end do
case(3)
 b=x;c=y;do d=0,nv-1;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;do a=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do;end do
case(4)
 b=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;do a=0,nv-1;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end do;end if;end do
case(5)
 d=x;e=y;q=adj(d*nv+e);if(q>0)then;do a=0,nv-1;do b=0,nv-1;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end do;end do;end if
case(6)
 a=x;b=y;do d=0,nv-1;do e=0,nv-1;q=adj(d*nv+e);c=d;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end do
case(7)
 a=x;b=y;do d=0,nv-1;e=d;q=adj(d*nv+e);if(q>0)then;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do
case(8)
 a=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;do b=0,nv-1;c=b;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do
case(9)
 a=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;b=e;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do
case(10)
 b=x;c=y;do d=0,nv-1;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;a=d;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end if;end do;end do
case(11)
 b=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;do a=0,nv-1;c=a;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do
case(12)
 b=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;a=e;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do
case(13)
 b=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;c=e;do a=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if;end do
case(14)
 d=x;e=y;q=adj(d*nv+e);if(q>0)then;do a=0,nv-1;b=a;do c=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end do;end if
case(15)
 d=x;e=y;q=adj(d*nv+e);if(q>0)then;do b=0,nv-1;c=b;do a=0,nv-1;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end do;end if
case(16)
 a=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;b=e;c=e;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end if;end do
case(17)
 b=x;d=y;do e=0,nv-1;q=adj(d*nv+e);if(q>0)then;a=e;c=e;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end if;end do
case(18)
 d=x;e=y;q=adj(d*nv+e);if(q>0)then;do a=0,nv-1;b=a;c=a;sum=sum+tensor(a*a0+b*b0+c*c0+f)*q;end do;end if
end select
