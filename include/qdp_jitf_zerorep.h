#ifndef QDP_JITFUNC_ZEROREP_H
#define QDP_JITFUNC_ZEROREP_H


namespace QDP {

  template<class T>
  void
  function_zero_rep_build( JitFunction& function, OLattice<T>& dest)
  {
    std::vector<ParamRef> params = jit_function_preamble_param("zero_rep",__PRETTY_FUNCTION__);

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

    llvm::Value * r_idx = jit_function_preamble_get_idx( params );

    zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    jit_get_function( function );
  }



  template<class T>
  void
  function_zero_rep_subtype_build( JitFunction& function, OSubLattice<T>& dest)
  {
    llvm_start_new_function("zero_rep_subtype",__PRETTY_FUNCTION__);

    ParamRef p_th_count     = llvm_add_param<int>();

    ParamLeaf param_leaf;
    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx_thread) );

    jit_get_function( function );
  }
  

  template<class T>
  void 
  function_zero_rep_exec(JitFunction& function, OLattice<T>& dest, const Subset& s )
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
    if (s.numSiteTable() < 1)
      return;

#ifdef QDP_DEEP_LOG
    function.start = s.start();
    function.count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
    function.size_T = sizeof(T);
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    function.set_dest_id( dest.getId() );
#endif
  
    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());

    int th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();

    JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
    JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
    JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
  
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_ordered.get_id() );
    ids.push_back( jit_th_count.get_id() );
    ids.push_back( jit_start.get_id() );
    ids.push_back( jit_end.get_id() );
    ids.push_back( s.getIdMemberTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }



  template<class T>
  void function_zero_rep_subtype_exec(JitFunction& function, OSubLattice<T>& dest, const Subset& s )
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());

    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }


  
} // QDP
#endif
