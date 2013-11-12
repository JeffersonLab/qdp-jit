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

  void jit_dispatch( void* function , int site_count, bool ordered, std::int64_t start, const AddressLeaf& args)
  {
    void (*FP)(std::int64_t,std::int64_t,std::int64_t,bool,std::int64_t,void*) = 
      (void (*)(std::int64_t,std::int64_t,std::int64_t,bool,std::int64_t,void*))(intptr_t)function;

    int threads_num;
    std::int64_t myId;
    std::int64_t lo = 0;
    std::int64_t hi = site_count;
    void * addr = args.addr.data();

#pragma omp parallel shared(site_count, threads_num, ordered, start, addr) private(myId, lo, hi) default(shared)
    {
      threads_num = omp_get_num_threads();
      myId = omp_get_thread_num();
      lo = site_count*myId/threads_num;
      hi = site_count*(myId+1)/threads_num;

      FP( lo , hi , myId , ordered, start, addr );
    }
  }



}

#else 

#if defined(QDP_USE_QMT_THREADS)
#error "QMT threading not implemented"
#endif

#error "Must use threading"

#endif

