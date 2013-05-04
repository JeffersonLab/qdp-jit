#include "qdp.h"

namespace QDP {

  jit_value_t datalayout( JitDeviceLayout lay , IndexDomainVector a ) {
    const size_t nIv = 0; // volume
    const size_t nIs = 1; // spin
    const size_t nIc = 2; // color
    const size_t nIr = 3; // reality

    int         Lv,Ls,Lc,Lr;
    jit_value_t iv,is,ic,ir;

    std::tie(Lv,iv) = a.at(nIv);
    std::tie(Ls,is) = a.at(nIs);
    std::tie(Lc,ic) = a.at(nIc);
    std::tie(Lr,ir) = a.at(nIr);

    jit_value_t Iv = create_jit_value(Lv);
    jit_value_t Is = create_jit_value(Ls);
    jit_value_t Ic = create_jit_value(Lc);
    jit_value_t Ir = create_jit_value(Lr);

    // offset = ((ir * Ic + ic) * Is + is) * Iv + iv

    if (lay == JitDeviceLayout::Coalesced)
      return jit_ins_add(jit_ins_mul(jit_ins_add(jit_ins_mul( jit_ins_add(jit_ins_mul(ir,Ic),ic),Is),is),Iv),iv);
    else
      return jit_ins_add(jit_ins_mul(jit_ins_add(jit_ins_mul( jit_ins_add(jit_ins_mul(iv,Ir),ir),Ic),ic),Is),is);
  }

} // namespace
