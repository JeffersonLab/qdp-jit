#ifndef QDP_JITF_GLOBMAX_H
#define QDP_JITF_GLOBMAX_H

namespace QDP {


template<class T>
void *
function_global_max_build(const OLattice<T>& src)
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  JitMainLoop loop;
  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t    TVIEW;
  typedef typename WordType<T>::Type_t                            TW;
  typedef typename REGType< typename JITType<T>::Type_t >::Type_t TREG;

  TVIEW src_view(forEach(src, param_leaf, TreeCombine()));

  OLatticeJIT<typename JITType<T>::Type_t > dest_jit( llvm_add_param< TW* >() );

  IndexDomainVector th_num = get_scalar_index_vector_from_index( loop.getThreadNum() );
  IndexDomainVector idx_lo = get_index_vector_from_index( loop.getLo() );

  TREG Treg_lo;
  Treg_lo.setup( src_view.elem( JitDeviceLayout::LayoutCoalesced , idx_lo ) );
  dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) = Treg_lo;

  IndexDomainVector idx    = loop.getIdx();

  // IndexDomainVector access = idx;
  // access[0].second = loop.getThreadNum();

  TREG Treg;
  Treg.setup( src_view.elem( JitDeviceLayout::LayoutCoalesced , idx ) );
  TREG dest_reg;
  dest_reg.setup( dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) );


  dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) = where( Treg > dest_reg , Treg , dest_reg );

  loop.done();

  return jit_function_epilogue_get("jit_globalmax.ptx");
}





#if 0
  llvm::BasicBlock * block_first = llvm_new_basic_block();
  llvm::BasicBlock * block_not_first = llvm_new_basic_block();
  llvm::BasicBlock * block_cont = llvm_new_basic_block();

  llvm::Value *isFirst = llvm_create_value(1);
  llvm_cond_branch( llvm_eq( isFirst , llvm_create_value(1) ) , block_first , block_not_first );

  llvm_set_insert_point( block_first );
  dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) = Treg;
  llvm_branch( block_cont );

  llvm_set_insert_point( block_not_first );
  dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) = where( Treg > dest_reg , Treg , dest_reg );
  llvm_branch( block_cont );

  llvm_set_insert_point( block_cont );
#endif



template<class T>
void 
function_global_max_exec(void * function, typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t& ret, const OLattice<T>& src, const Subset& s)
{
  typedef typename UnaryReturn<T, FnGlobalMax>::Type_t RetT;

  // I cannot initialize 'dest' here. This must be done in the kernel
  RetT dest[ qdpNumThreads() ];

  AddressLeaf addr_leaf;

  int junk_src = forEach(src, addr_leaf, NullCombine());
  addr_leaf.setAddr( &dest[0] );

#ifdef LLVM_DEBUG
  std::cout << "calling globalMax(Lattice).. " << addr_leaf.addr.size() << "\n";
#endif

  jit_dispatch( function , s.numSiteTable() , s.hasOrderedRep() , s.start() , addr_leaf );

  ret.elem() = dest[0];
  for( int i = 1 ; i < qdpNumThreads() ; ++i )
    if (toBool(dest[i] > ret.elem()))
      ret.elem() = dest[i];

  // MPI globalMax in caller
}



}
#endif
