#ifndef QDP_AUTOTUNING_H
#define QDP_AUTOTUNING_H

namespace QDP {

  void jit_call(void* function,int th_count,const AddressLeaf& args);

  int jit_autotuning(void* function,int lo,int hi,void ** param);

}

#endif
