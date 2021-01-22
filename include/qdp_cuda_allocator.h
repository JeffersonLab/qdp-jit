// -*- C++ -*-

/*! \file
 * \brief Cuda memory allocator for QDP
 *
 */



#ifndef QDP_CUDA_ALLOCATOR
#define QDP_CUDA_ALLOCATOR

namespace QDP
{

  class QDPCUDAAllocator {
  public:
    enum { ALIGNMENT_SIZE = 4096 };

    static bool allocate( void** ptr, const size_t n_bytes ) {
      return gpu_malloc( ptr , n_bytes );
    }

    static void free(const void *mem) {
      gpu_free( mem );
    }
  };


} // namespace QDP



#endif


