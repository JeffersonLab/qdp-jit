#ifndef QDP_DISPATCH_H
#define QDP_DISPATCH_H

#include <omp.h>

namespace QDP { 

  int qdpNumThreads();
  void qdpSetNumThreads(int);
  void jit_dispatch( void* function , int site_count, bool ordered, int64_t start, const AddressLeaf& args);


  template<class Arg>
  void dispatch_to_threads(int numSiteTable, Arg a, void (*func)(int,int,int, Arg*))
  {
    int threads_num;
    int myId;
    int low = 0;
    int high = numSiteTable;
   
#pragma omp parallel shared(numSiteTable, threads_num, a) private(myId, low, high) default(shared)
    {
     
      threads_num = omp_get_num_threads();
      myId = omp_get_thread_num();
      low = numSiteTable*myId/threads_num;
      high = numSiteTable*(myId+1)/threads_num;
 
      func(low, high, myId, &a);
    }
  }


}

#endif
