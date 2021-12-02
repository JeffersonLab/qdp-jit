#ifndef QDP_JITFUNC_EXTRACT_H
#define QDP_JITFUNC_EXTRACT_H


namespace QDP {


  template<class T , class T2>
  inline void 
  function_extract_exec(JitFunction& function, multi1d<OScalar<T> >& dest, const OLattice<T2>& src, const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    //function.set_dest_id( dest.getId() );
    function.set_is_lat(false);
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

    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
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

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    OLatticeJIT< typename JITType<T >::Type_t > odata  ( llvm_add_param< typename WordType<T >::Type_t* >());

    OLatticeJIT< typename JITType< typename ScalarType<T2>::Type_t >::Type_t > src_jit( llvm_add_param< typename WordType<T2>::Type_t* >());

    
    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType<T>::Type_t >::Type_t in_data_reg;

#if defined(QDP_CODEGEN_VECTOR)
    in_data_reg.setup( src_jit.elem( JitDeviceLayout::Coalesced_scalar_idx , r_idx ) );
#else
    in_data_reg.setup( src_jit.elem( JitDeviceLayout::Coalesced            , r_idx ) );
#endif
    
    odata.elem( JitDeviceLayout::Scalar , r_idx_thread ) = in_data_reg;

    jit_get_function( function );
  }



  
} // QDP
#endif
