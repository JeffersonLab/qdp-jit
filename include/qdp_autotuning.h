#ifndef QDP_AUTOTUNING_H
#define QDP_AUTOTUNING_H

namespace QDP {

  void jit_launch(JitFunction function,int th_count,std::vector<QDPCache::ArgKey>& ids);

  //int jit_autotuning(JitFunction function,int lo,int hi,void ** param);

}

#endif
