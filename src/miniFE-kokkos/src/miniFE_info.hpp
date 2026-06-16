#ifndef miniFE_info_hpp
#define miniFE_info_hpp

#define MINIFE_HOSTNAME "copilot-gpu-sandbox"
#define MINIFE_KERNEL_NAME "'Linux'"
#define MINIFE_KERNEL_RELEASE "'6.14.0-1015-nvidia'"
#define MINIFE_PROCESSOR "'aarch64'"

#define MINIFE_CXX "'/usr/bin/g++'"
#define MINIFE_CXX_VERSION "'g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0'"
#define MINIFE_CXXFLAGS "' -std=c++17 -fopenmp -I/home/copilot/kokkos-install/include -Isrc -Iutils -Ifem -DMINIFE_SCALAR=double -DMINIFE_LOCAL_ORDINAL=int -DMINIFE_GLOBAL_ORDINAL=int -DMINIFE_RESTRICT=__restrict__ -DMINIFE_CSR_MATRIX -DMINIFE_INFO=1 -DMINIFE_KERNELS=0 -O3'"

#endif
