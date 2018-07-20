#ifndef QDP_AUTOTUNING_H
#define QDP_AUTOTUNING_H

namespace QDP {

  void jit_launch(CUfunction function,int th_count,std::vector<int>& ids);

  //int jit_autotuning(CUfunction function,int lo,int hi,void ** param);

}

#endif
