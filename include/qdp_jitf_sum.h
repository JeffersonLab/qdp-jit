#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

namespace QDP {



template<class T>
void 
function_sum_build( JitFunction& func, const OLattice<T>& src)
{
#ifdef LLVM_DEBUG
  QDPIO::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  if (llvm_debug::debug_func_write) {
    if (Layout::primaryNode()) {
      llvm_debug_write_set_name(__PRETTY_FUNCTION__,"");
    }
  }

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t    TVIEW;
  typedef typename UnaryReturn<T, FnSum>::Type_t                  TUP;
  typedef typename WordType<TUP>::Type_t                          TUPW;
  typedef typename REGType< typename JITType<T>::Type_t >::Type_t TREG;

  {
    JitMainLoop loop;
    ParamLeaf param_leaf;

    TVIEW src_view(forEach(src, param_leaf, TreeCombine()));

    OLatticeJIT<typename JITType<TUP>::Type_t > dest_jit( llvm_add_param< TUPW* >() );

    IndexDomainVector idx    = loop.getIdx();
    IndexDomainVector th_num = get_scalar_index_vector_from_index( loop.getThreadNum() );

    TREG Treg;
    Treg.setup( src_view.elem( JitDeviceLayout::LayoutCoalesced , idx ) );

    dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) += Treg;

    loop.done();

    func.func().push_back( jit_function_epilogue_get("jit_sum.ptx") );
  }

  {
    JitMainLoop loop( 1 , true );
    ParamLeaf param_leaf;

    TVIEW src_view(forEach(src, param_leaf, TreeCombine()));

    OLatticeJIT<typename JITType<TUP>::Type_t > dest_jit( llvm_add_param< TUPW* >() );

    IndexDomainVector idx    = loop.getIdx();
    IndexDomainVector th_num = get_scalar_index_vector_from_index( loop.getThreadNum() );

    TREG Treg;
    Treg.setup( src_view.elem( JitDeviceLayout::LayoutCoalesced , idx ) );

    dest_jit.elem( JitDeviceLayout::LayoutScalar , th_num ) += Treg;

    loop.done();

    func.func().push_back( jit_function_epilogue_get("jit_sum.ptx") );
  }
}




template<class T>
void 
function_sum_exec(const JitFunction& function, typename UnaryReturn<OLattice<T>, FnSum>::Type_t& ret, const OLattice<T>& src, const Subset& s)
{
#ifdef LLVM_DEBUG
  QDPIO::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  //assert( s.hasOrderedRep() );

  typedef typename UnaryReturn<T, FnSum>::Type_t RetT;

  RetT* dest = new RetT[ qdpNumThreads() ];

  for( int i = 0 ; i < qdpNumThreads() ; ++i )
    zero_rep(dest[i]);

  AddressLeaf addr_leaf(s);

  int inner    = getDataLayoutInnerSize();
  int func_num = 0;
  
  if( !s.hasOrderedRep() ) {
    //QDPIO::cout << "jit_sum_exec: subset has no ordered set\n";
    AddressLeaf::Types t;
    t.ptr = const_cast<int*>( s.siteTable().slice() );
    addr_leaf.addr.push_back(t);
    inner = 1;
    func_num = 1;
  }
  
  forEach(src, addr_leaf, NullCombine());
  addr_leaf.setAddr( dest );

#ifdef LLVM_DEBUG
  std::cout << "calling sum(Lattice).. " << addr_leaf.addr.size() << "\n";
#endif

  if (s.numSiteTable() % getDataLayoutInnerSize())
    QDP_error_exit("number of sites in ordered subset is %d, but inner length is %d" , 
		   s.numSiteTable() , getDataLayoutInnerSize());

  jit_dispatch( function.func().at( func_num ) , s.numSiteTable() , inner , s.hasOrderedRep() , s.start() , addr_leaf );

  
  zero_rep(ret);
  for( int i = 0 ; i < qdpNumThreads() ; ++i )
    ret.elem() += dest[i];

  delete[] dest;



  
  // MPI sum in caller
}



} // namespace

#endif
