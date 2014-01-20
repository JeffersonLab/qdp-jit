#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

namespace QDP {



template<class T>
void 
function_sum_build( JitFunction& func, const OLattice<T>& src)
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
  Treg.setup( src_view.elem( JitDeviceLayout::LayoutCoalesced , idx ) );

  dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) += Treg;

  loop.done();

  func.func().push_back( jit_function_epilogue_get("jit_sum.ptx") );
}




template<class T>
void 
function_sum_exec(const JitFunction& function, typename UnaryReturn<OLattice<T>, FnSum>::Type_t& ret, const OLattice<T>& src, const Subset& s)
{
  typedef typename UnaryReturn<T, FnSum>::Type_t RetT;

  RetT* dest = new RetT( qdpNumThreads() );

  for( int i = 0 ; i < qdpNumThreads() ; ++i )
    zero_rep(dest[i]);

  AddressLeaf addr_leaf(s);

  int junk_src = forEach(src, addr_leaf, NullCombine());
  addr_leaf.setAddr( dest );

#ifdef LLVM_DEBUG
  std::cout << "calling sum(Lattice).. " << addr_leaf.addr.size() << "\n";
#endif

  jit_dispatch( function.func().at(0) , s.numSiteTable() , s.hasOrderedRep() , s.start() , addr_leaf );

  zero_rep(ret);
  for( int i = 0 ; i < qdpNumThreads() ; ++i )
    ret.elem() += dest[i];

  delete[] dest;

  // MPI sum in caller
}



} // namespace

#endif
