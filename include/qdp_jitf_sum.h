#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

#include "qmp.h"

namespace QDP {



template<class T>
void *
function_sum_build(const OLattice<T>& src)
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  JitMainLoop loop;
  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t    TVIEW;
  typedef typename UnaryReturn<T, FnSum>::Type_t                  TUP;
  typedef typename WordType<TUP>::Type_t                          TUPW;
  typedef typename REGType< typename JITType<T>::Type_t >::Type_t TREG;

  TVIEW src_view(forEach(src, param_leaf, TreeCombine()));

  OLatticeJIT<typename JITType<TUP>::Type_t > dest_jit( llvm_add_param< TUPW* >() );

  IndexDomainVector idx    = loop.getIdx();
  IndexDomainVector th_num = get_scalar_index_vector_from_index( loop.getThreadNum() );

  // IndexDomainVector access = idx;
  // access[0].second = loop.getThreadNum();

  TREG Treg;
  Treg.setup( src_view.elem( JitDeviceLayout::Coalesced , idx ) );

  dest_jit.elem( JitDeviceLayout::Scalar , th_num ) += Treg;

  loop.done();

  return jit_function_epilogue_get("jit_sum.ptx");
}




template<class T>
void 
function_sum_exec(void * function, typename UnaryReturn<OLattice<T>, FnSum>::Type_t& ret, const OLattice<T>& src, const Subset& s)
{
  typedef typename UnaryReturn<T, FnSum>::Type_t RetT;

  RetT dest[ 32 * getDataLayoutInnerSize() ];

  for( int i = 0 ; i < 32 ; ++i )
    zero_rep(dest[i]);

  AddressLeaf addr_leaf;
  jit_get_empty_arguments(addr_leaf);

  addr_leaf.setOrdered( s.hasOrderedRep() );
  addr_leaf.setStart( s.start() );

  int junk_src = forEach(src, addr_leaf, NullCombine());
  addr_leaf.setAddr( &dest[0] );

#ifdef LLVM_DEBUG
#endif
  std::cout << "calling sum(Lattice).. " << addr_leaf.addr.size() << "\n";

  jit_call( function , s.numSiteTable() , addr_leaf );

  zero_rep(ret);
  for( int i = 0 ; i < 32 ; ++i )
    ret.elem() += dest[i];
}



} // namespace

#endif
