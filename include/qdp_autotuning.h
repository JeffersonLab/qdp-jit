#ifndef QDP_AUTOTUNING_H
#define QDP_AUTOTUNING_H

namespace QDP {

  int jit_autotuning(CUfunction function,int lo,int hi,void ** param);

}

#endif
