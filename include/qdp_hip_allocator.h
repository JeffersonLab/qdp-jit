// -*- C++ -*-

/*! \file
 * \brief Cuda memory allocator for QDP
 *
 */



#ifndef QDP_HIP_ALLOCATOR
#define QDP_HIP_ALLOCATOR

namespace QDP
{

  class QDPHIPAllocator {
  public:
    enum { ALIGNMENT_SIZE = 4096 };

    static bool allocate( void** ptr, const size_t n_bytes ) {
      return HipMalloc( ptr , n_bytes );
    }

    static void free(const void *mem) {
      HipFree( mem );
    }
  };


} // namespace QDP



#endif


