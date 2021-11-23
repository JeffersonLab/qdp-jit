#ifndef QDP_JITFUNC_EXTRACT_H
#define QDP_JITFUNC_EXTRACT_H


namespace QDP {


  template<class T , class T2>
  inline void 
  function_extract_exec(JitFunction& function, multi1d<OScalar<T> >& dest, const OLattice<T2>& src, const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    // function.start = s.start();
    // function.count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
    // function.size_T = sizeof(T);
    // function.type_W = typeid(typename WordType<T>::Type_t).name();
    // function.set_dest_id( dest.getId() );
#endif
    
    if (s.numSiteTable() < 1)
      return;

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(T) * s.numSiteTable() , dest.slice() , nullptr );

    AddressLeaf addr_leaf(s);
    forEach(src, addr_leaf, NullCombine());

    // // For tuning
    // function.set_dest_id( dest.getId() );
    // function.set_enable_tuning();
  
    int th_count = s.numSiteTable();
		
    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    ids.push_back( s.getIdSiteTable() );
    ids.push_back( d_id );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
	  
    jit_launch(function,th_count,ids);

    // need to sync GPU
    jit_util_sync_copy();
    
    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );
  }

  
  

  template<class T , class T2>
  inline void 
  function_extract_build(JitFunction& function, multi1d<OScalar<T> >& dest, const OLattice<T2>& src)
  {
    llvm_start_new_function("extract", __PRETTY_FUNCTION__ );

    ParamRef p_th_count   = llvm_add_param<int>();
    ParamRef p_site_table = llvm_add_param<int*>();

    typedef typename WordType<T>::Type_t TWT;
    ParamRef p_odata      = llvm_add_param< TWT* >();  // output array

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T2>, ParamLeaf>::Type_t  FuncRet_t;
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));

    OLatticeJIT<typename JITType<T>::Type_t> odata( p_odata );   // want scalar access later

    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType<T>::Type_t >::Type_t in_data_reg;   
    in_data_reg.setup( src_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );
      
    odata.elem( JitDeviceLayout::Scalar , r_idx_thread ) = in_data_reg;

    jit_get_function( function );
  }

} // QDP
#endif
