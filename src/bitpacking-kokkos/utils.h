#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <climits>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <Kokkos_Core.hpp>

constexpr int const BLOCK_SIZE = 256;
constexpr int const BLOCK_WIDTH = 4096;

typedef enum nvcompType_t
{
  NVCOMP_TYPE_CHAR = 0,
  NVCOMP_TYPE_UCHAR = 1,
  NVCOMP_TYPE_SHORT = 2,
  NVCOMP_TYPE_USHORT = 3,
  NVCOMP_TYPE_INT = 4,
  NVCOMP_TYPE_UINT = 5,
  NVCOMP_TYPE_LONGLONG = 6,
  NVCOMP_TYPE_ULONGLONG = 7,
  NVCOMP_TYPE_BITS = 0xff
} nvcompType_t;

template <typename T>
T unpackBytes(
    const void* data, const uint8_t numBits, const T minValue, const size_t i)
{
  using U = typename std::make_unsigned<T>::type;

  if (numBits == 0) {
    return minValue;
  } else {
    uint8_t scratch[9] = {0,0,0,0,0,0,0,0,0};
    const U mask = numBits < sizeof(T)*8U
        ? static_cast<U>((1ULL << numBits) - 1)
        : static_cast<U>(-1);
    const uint8_t* byte_data = reinterpret_cast<decltype(byte_data)>(data);

    size_t start_byte = (i * numBits) / 8;
    size_t end_byte = ((i + 1) * numBits - 1) / 8;
    assert(end_byte - start_byte <= sizeof(scratch));

    for (size_t j = start_byte, k = 0; j <= end_byte; ++j, ++k)
      scratch[k] = byte_data[j];

    const int bitOffset = (i * numBits) % 8;
    U baseValue = 0;
    for (size_t k = 0; k <= end_byte - start_byte; ++k) {
      U shifted;
      if (k > 0)
        shifted = static_cast<U>(scratch[k]) << ((k * 8) - bitOffset);
      else
        shifted = static_cast<U>(scratch[k]) >> bitOffset;
      baseValue |= mask & shifted;
    }
    return baseValue + minValue;
  }
}

size_t getReduceScratchSpaceSize(size_t const num);
size_t requiredWorkspaceSize(size_t const num, const nvcompType_t type);

template <typename T>
inline nvcompType_t TypeOf()
{
  if (std::is_same<T, int8_t>::value)        return NVCOMP_TYPE_CHAR;
  else if (std::is_same<T, uint8_t>::value)  return NVCOMP_TYPE_UCHAR;
  else if (std::is_same<T, int16_t>::value)  return NVCOMP_TYPE_SHORT;
  else if (std::is_same<T, uint16_t>::value) return NVCOMP_TYPE_USHORT;
  else if (std::is_same<T, int32_t>::value)  return NVCOMP_TYPE_INT;
  else if (std::is_same<T, uint32_t>::value) return NVCOMP_TYPE_UINT;
  else if (std::is_same<T, int64_t>::value)  return NVCOMP_TYPE_LONGLONG;
  else if (std::is_same<T, uint64_t>::value) return NVCOMP_TYPE_ULONGLONG;
  else return NVCOMP_TYPE_INT;
}

inline size_t sizeOfnvcompType(nvcompType_t type)
{
  switch (type) {
  case NVCOMP_TYPE_BITS:      return 1;
  case NVCOMP_TYPE_CHAR:      return sizeof(int8_t);
  case NVCOMP_TYPE_UCHAR:     return sizeof(uint8_t);
  case NVCOMP_TYPE_SHORT:     return sizeof(int16_t);
  case NVCOMP_TYPE_USHORT:    return sizeof(uint16_t);
  case NVCOMP_TYPE_INT:       return sizeof(int32_t);
  case NVCOMP_TYPE_UINT:      return sizeof(uint32_t);
  case NVCOMP_TYPE_LONGLONG:  return sizeof(int64_t);
  case NVCOMP_TYPE_ULONGLONG: return sizeof(uint64_t);
  default:
    throw std::runtime_error("Unsupported type " + std::to_string(type));
  }
}

template <typename U, typename T>
KOKKOS_INLINE_FUNCTION U roundUpDiv(U const num, T const chunk)
{
  return (num / chunk) + (num % chunk > 0);
}

template <typename U, typename T>
KOKKOS_INLINE_FUNCTION U roundDownTo(U const num, T const chunk)
{
  return (num / chunk) * chunk;
}

template <typename U, typename T>
KOKKOS_INLINE_FUNCTION U roundUpTo(U const num, T const chunk)
{
  return roundUpDiv(num, chunk) * chunk;
}

// Portable count-leading-zeros for uint32_t, usable on host and device.
KOKKOS_INLINE_FUNCTION int countLeadingZeros32(uint32_t x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __clz(x);
#else
  if (x == 0) return 32;
  return __builtin_clz(x);
#endif
}

void compress(
    Kokkos::View<int32_t*> minScratch,
    Kokkos::View<int32_t*> maxScratch,
    Kokkos::View<uint32_t*> output,
    Kokkos::View<int32_t*>  input,
    size_t n,
    Kokkos::View<int32_t*>       minValueDevice,
    Kokkos::View<unsigned char*> numBitsDevice);

#endif
