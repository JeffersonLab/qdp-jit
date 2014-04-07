// -*- C++ -*-

#ifndef QDP_DATALAYOUT_H
#define QDP_DATALAYOUT_H


namespace QDP {

  void QDP_set_jit_datalayout(int pos_o, int pos_s, int pos_c, int pos_r, int pos_i);
  void QDP_print_jit_datalayout();
  //llvm::Value * datalayout_stack(IndexDomainVector a);

} // namespace QDP

#endif
