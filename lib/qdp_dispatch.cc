#include "qdp.h"

#if defined(QDP_USE_OMP_THREADS)

#include <omp.h>
namespace QDP {    
  
  int qdpNumThreads() {
    return omp_get_max_threads();
  }

  void qdpSetNumThreads(int n) {
    omp_set_num_threads(n);
  }

  void jit_dispatch( void* function , int site_count, const AddressLeaf& args)
  {
    void (*FP)(void*) = (void (*)(void*))(intptr_t)function;

    int threads_num;
    std::int64_t myId;
    std::int64_t lo = 0;
    std::int64_t hi = site_count;
    AddressLeaf my_args;

#pragma omp parallel shared(site_count, threads_num, args) private(myId, lo, hi, my_args) default(shared)
    {
      threads_num = omp_get_num_threads();
      myId = omp_get_thread_num();
      lo = site_count*myId/threads_num;
      hi = site_count*(myId+1)/threads_num;

      my_args = args;
      my_args.addr[0].in64 = lo;
      my_args.addr[1].in64 = hi;
      my_args.addr[2].in64 = myId;

      FP( my_args.addr.data() );
#pragma omp barrier
    }
  }



}

#else 

#if defined(QDP_USE_QMT_THREADS)
#error "QMT threading not implemented"
#endif

#error "Must use threading"

#endif

