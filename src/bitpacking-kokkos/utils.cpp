#include "utils.h"

// Returns the number of bytes needed for one min- or max-scratch array.
// The value is used both as a byte count and (in the original code) as a
// pointer offset expressed in elements of the working type; the sizeof(uint64_t)
// factor guarantees sufficient space and 8-byte alignment for any supported type.
size_t getReduceScratchSpaceSize(size_t const num)
{
  size_t const base =
      std::min(BLOCK_WIDTH, static_cast<int>(roundUpDiv(num, (size_t)BLOCK_SIZE)))
      * sizeof(uint64_t);
  return base;
}

// Total workspace bytes required to compress `num` elements of `type`.
// Two scratch arrays (min and max) are needed.
size_t requiredWorkspaceSize(size_t const num, const nvcompType_t type)
{
  return sizeOfnvcompType(type) * getReduceScratchSpaceSize(num) * 2;
}
