#ifndef QDP_DISPATCH_H
#define QDP_DISPATCH_H

namespace QDP { 

  int qdpNumThreads();
  void qdpSetNumThreads(int);
  void jit_dispatch( void* function , int site_count, bool ordered, std::int64_t start, const AddressLeaf& args);

}

#endif
