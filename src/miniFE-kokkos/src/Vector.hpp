#ifndef _Vector_hpp_
#define _Vector_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// ************************************************************************
//@HEADER

#include <vector>
#include <Kokkos_Core.hpp>

namespace miniFE {


template<typename Scalar,
         typename LocalOrdinal,
         typename GlobalOrdinal>
struct Vector {
  typedef Scalar ScalarType;
  typedef LocalOrdinal LocalOrdinalType;
  typedef GlobalOrdinal GlobalOrdinalType;

  Vector(GlobalOrdinal startIdx, LocalOrdinal local_sz)
   : startIndex(startIdx),
     local_size(local_sz),
     coefs(local_size, 0),
     d_coefs("vec_d_coefs", local_sz)
  {
    Kokkos::deep_copy(d_coefs, static_cast<Scalar>(0));
  }

  ~Vector()
  {
  }

  void sync_to_device() {
    auto h = Kokkos::create_mirror_view(d_coefs);
    for (size_t i = 0; i < coefs.size(); ++i) h(i) = coefs[i];
    Kokkos::deep_copy(d_coefs, h);
  }

  void sync_to_host() {
    auto h = Kokkos::create_mirror_view(d_coefs);
    Kokkos::deep_copy(h, d_coefs);
    for (size_t i = 0; i < coefs.size(); ++i) coefs[i] = h(i);
  }

  GlobalOrdinal startIndex;
  LocalOrdinal local_size;
  std::vector<Scalar> coefs;
  mutable Kokkos::View<Scalar*> d_coefs;
};


}//namespace miniFE

#endif

