#ifndef QDP_JITF_GAUSS_H
#define QDP_JITF_GAUSS_H

#include "qmp.h"

namespace QDP {

template<class T>
void *
function_gaussian_build(OLattice<T>& dest ,OLattice<T>& r1 ,OLattice<T>& r2 )
{
  JitMainLoop loop;

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;

  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
  FuncRet_t r1_jit(forEach(r1, param_leaf, TreeCombine()));
  FuncRet_t r2_jit(forEach(r2, param_leaf, TreeCombine()));

  IndexDomainVector idx = loop.getIdx();

  typedef typename REGType< typename JITType<T>::Type_t >::Type_t TREG;

  TREG r1_reg;
  TREG r2_reg;
  r1_reg.setup( r1_jit.elem( JitDeviceLayout::Coalesced , idx ) );
  r2_reg.setup( r2_jit.elem( JitDeviceLayout::Coalesced , idx ) );

  fill_gaussian( dest_jit.elem(JitDeviceLayout::Coalesced , idx ) , r1_reg , r2_reg );

  loop.done();

  return jit_function_epilogue_get("jit_gaussian.ptx");
}


template<class T>
void 
function_gaussian_exec(void *function, OLattice<T>& dest,OLattice<T>& r1,OLattice<T>& r2, const Subset& s )
{
  assert( s.hasOrderedRep() );

  AddressLeaf addr_leaf;
  jit_get_empty_arguments(addr_leaf);

  int junk_0 = forEach(dest, addr_leaf, NullCombine());
  int junk_1 = forEach(r1, addr_leaf, NullCombine());
  int junk_2 = forEach(r2, addr_leaf, NullCombine());

  addr_leaf.setOrdered( s.hasOrderedRep() );
  addr_leaf.setStart( s.start() );

  jit_dispatch(function,s.numSiteTable(),addr_leaf);
}



}

#endif
