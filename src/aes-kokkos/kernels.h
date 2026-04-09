#ifndef KERNELS_H
#define KERNELS_H

#include <Kokkos_Core.hpp>
#include "aes.h"

KOKKOS_INLINE_FUNCTION
uchar galoisMultiplication(uchar a, uchar b)
{
  uchar p = 0; 
  for(unsigned int i=0; i < 8; ++i)
  {
    if((b&1) == 1)
    {
      p^=a;
    }
    uchar hiBitSet = (a & 0x80);
    a <<= 1;
    if(hiBitSet == 0x80)
    {
      a ^= 0x1b;
    }
    b >>= 1;
  }
  return p;
}

KOKKOS_INLINE_FUNCTION
uchar4 sboxRead(const uchar * SBox, uchar4 block)
{
  return uchar4(SBox[block.x], SBox[block.y], SBox[block.z], SBox[block.w]);
}

KOKKOS_INLINE_FUNCTION
uchar4 mixColumnsDevice(const uchar4 * block, const uchar4 * galiosCoeff, unsigned int j)
{
  unsigned int bw = 4;

  uchar x, y, z, w;

  x = galoisMultiplication(block[0].x, galiosCoeff[(bw-j)%bw].x);
  y = galoisMultiplication(block[0].y, galiosCoeff[(bw-j)%bw].x);
  z = galoisMultiplication(block[0].z, galiosCoeff[(bw-j)%bw].x);
  w = galoisMultiplication(block[0].w, galiosCoeff[(bw-j)%bw].x);

  for(unsigned int k=1; k< 4; ++k)
  {
    x ^= galoisMultiplication(block[k].x, galiosCoeff[(k+bw-j)%bw].x);
    y ^= galoisMultiplication(block[k].y, galiosCoeff[(k+bw-j)%bw].x);
    z ^= galoisMultiplication(block[k].z, galiosCoeff[(k+bw-j)%bw].x);
    w ^= galoisMultiplication(block[k].w, galiosCoeff[(k+bw-j)%bw].x);
  }

  return uchar4(x, y, z, w);
}

KOKKOS_INLINE_FUNCTION
uchar4 shiftRowsDevice(uchar4 row, unsigned int j)
{
  uchar4 r = row;
  for(uint i=0; i < j; ++i)  
  {
    uchar x = r.x;
    uchar y = r.y;
    uchar z = r.z;
    uchar w = r.w;
    r = uchar4(y,z,w,x);
  }
  return r;
}

KOKKOS_INLINE_FUNCTION
uchar4 shiftRowsInvDevice(uchar4 row, unsigned int j)
{
  uchar4 r = row;
  for(uint i=0; i < j; ++i)  
  {
    uchar x = r.x;
    uchar y = r.y;
    uchar z = r.z;
    uchar w = r.w;
    r = uchar4(w,x,y,z);
  }
  return r;
}

#endif // KERNELS_H
