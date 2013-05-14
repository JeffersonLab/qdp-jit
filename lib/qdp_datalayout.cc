#include "qdp.h"

namespace QDP {

#if 0
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

    if (lay == JitDeviceLayout::Coalesced) {
      return jit_ins_add(jit_ins_mul(jit_ins_add(jit_ins_mul( jit_ins_add(jit_ins_mul(ir,Ic),ic),Is),is),Iv),iv);
    } else
      return jit_ins_add(jit_ins_mul(jit_ins_add(jit_ins_mul( jit_ins_add(jit_ins_mul(iv,Ir),ir),Ic),ic),Is),is);
  }
#endif


#if 0
  jit_value_t datalayout_stack( JitDeviceLayout lay , IndexDomainVector a ) {
    assert(a.size() > 0);
    jit_value_t offset = create_jit_value(0);
    for( auto x = a.rbegin() ; x != a.rend() ; x++ ) {
      int         Index;
      jit_value_t index;
      std::tie(Index,index) = *x;
      jit_value_t Index_jit = create_jit_value(Index);
      offset = jit_ins_add( jit_ins_mul( offset , Index_jit ) , index );
    }
    return offset;
  }
#endif


  jit_value_t datalayout( JitDeviceLayout lay , IndexDomainVector a ) {
    assert(a.size() > 0);

    // In case of a coalesced layout (OLattice)
    // We reverse the data layout given by the natural nesting order
    // of aggregates, i.e. reality slowest, lattice fastest
    // In case of a scalar layout (sums,comms buffers,OScalar)
    // We actually use the index order/data layout given by the
    // nesting order of aggregates
    if ( lay == JitDeviceLayout::Coalesced ) {
      std::reverse( a.begin() , a.end() );
    }

    jit_value_t offset = create_jit_value(0);
    for( auto x = a.begin() ; x != a.end() ; x++ ) {
      int         Index;
      jit_value_t index;
      std::tie(Index,index) = *x;
      jit_value_t Index_jit = create_jit_value(Index);
      offset = jit_ins_add( jit_ins_mul( offset , Index_jit ) , index );
    }
    return offset;
  }



} // namespace
