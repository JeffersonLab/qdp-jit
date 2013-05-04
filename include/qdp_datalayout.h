// -*- C++ -*-

#ifndef QDP_DATALAYOUT_H
#define QDP_DATALAYOUT_H

namespace QDP {

  typedef std::pair< int , jit_value_t > IndexDomain;
  typedef std::vector< IndexDomain >     IndexDomainVector;


  jit_value_t datalayout( JitDeviceLayout lay , IndexDomainVector a ) {
    const size_t nIv = 0; // volume
    const size_t nIs = 1; // spin
    const size_t nIc = 2; // color
    const size_t nIr = 3; // reality

    int         Iv,Is,Ic,Ir;
    jit_value_t lv,ls,lc,lr;

    std::tie(Iv,lv) = a.at(nIv);
    std::tie(Is,ls) = a.at(nIs);
    std::tie(Ic,lc) = a.at(nIc);
    std::tie(Ir,lr) = a.at(nIr);

    jit_value_t iv = create_jit_value(lv);
    jit_value_t is = create_jit_value(ls);
    jit_value_t ic = create_jit_value(lc);
    jit_value_t ir = create_jit_value(lr);

    // offset = ((ir * Ic + ic) * Is + is) * Iv + iv

    if (lay == JitDeviceLayout::coalesced)
      return jit_ins_add(jit_ins_mul(jit_ins_add(jit_ins_mul( jit_ins_add(jit_ins_mul(ir,Ic),ic),Is),is),Iv),iv);
    else
      return jit_ins_add(jit_ins_mul(jit_ins_add(jit_ins_mul( jit_ins_add(jit_ins_mul(iv,Ir),ir),Ic),ic),Is),is);

  }




} // namespace QDP

#endif
