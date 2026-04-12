/* ---------------------------------------------
Makefile constructed configuration:
----------------------------------------------*/
#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif

#define KOKKOS_VERSION 30702

/* Execution Spaces */
#define KOKKOS_ENABLE_OPENMP
/* General Settings */
#define KOKKOS_ENABLE_DEPRECATED_CODE_3
#define KOKKOS_ENABLE_CXX14
#define KOKKOS_ENABLE_COMPLEX_ALIGN
#define KOKKOS_ENABLE_LIBDL
/* Optimization Settings */
/* Cuda Settings */
#define KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
