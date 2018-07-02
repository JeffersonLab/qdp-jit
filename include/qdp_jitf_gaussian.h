#ifndef QDP_JITF_GAUSS_H
#define QDP_JITF_GAUSS_H

#include "qmp.h"

namespace QDP {

template<class T>
void 
function_gaussian_build( JitFunction& func, OLattice<T>& dest ,OLattice<T>& r1 ,OLattice<T>& r2 )
{
#ifdef LLVM_DEBUG
  QDPIO::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  if (llvm_debug::debug_func_write) {
    if (Layout::primaryNode()) {
      llvm_debug_write_set_name(__PRETTY_FUNCTION__,"");
    }
  }

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
  r1_reg.setup( r1_jit.elem( JitDeviceLayout::LayoutCoalesced , idx ) );
  r2_reg.setup( r2_jit.elem( JitDeviceLayout::LayoutCoalesced , idx ) );

  fill_gaussian( dest_jit.elem(JitDeviceLayout::LayoutCoalesced , idx ) , r1_reg , r2_reg );

  loop.done();

  func.func().push_back( jit_function_epilogue_get("jit_gaussian.ptx") );
}


template<class T>
void 
function_gaussian_exec(const JitFunction& function, OLattice<T>& dest,OLattice<T>& r1,OLattice<T>& r2, const Subset& s )
{
#ifdef LLVM_DEBUG
  QDPIO::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  assert( s.hasOrderedRep() );

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  forEach(r1, addr_leaf, NullCombine());
  forEach(r2, addr_leaf, NullCombine());

  if (s.numSiteTable() % getDataLayoutInnerSize())
    QDP_error_exit("number of sites in ordered subset is %d, but inner length is %d" , 
		   s.numSiteTable() , getDataLayoutInnerSize());

  jit_dispatch(function.func().at(0),s.numSiteTable(),getDataLayoutInnerSize(),s.hasOrderedRep(),s.start(),addr_leaf);
}



}

#endif
