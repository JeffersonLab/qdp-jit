#ifndef QDP_JITFUNC_ZEROREP_H
#define QDP_JITFUNC_ZEROREP_H


namespace QDP {

  template<class T>
  void
  function_zero_rep_build( JitFunction& function, OLattice<T>& dest)
  {
    llvm_start_new_function("zero_rep",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  destJIT_t;
    destJIT_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );
    
    zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    jit_get_function( function );
  }



  template<class T>
  void
  function_zero_rep_subtype_build( JitFunction& function, OSubLattice<T>& dest)
  {
    llvm_start_new_function("zero_rep_subtype",__PRETTY_FUNCTION__);
    
    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeaf param_leaf;
    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  
    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );
    
    zero_rep( dest_jit.elemScalar(JitDeviceLayout::Coalesced,r_idx) );

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
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    function.set_dest_id( dest.getId() );
    function.set_is_lat(true);
#endif
  
    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());

    int th_count = s.numSiteTable();

    WorkgroupGuardExec workgroupGuardExec(th_count);
  
    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);

#ifdef QDP_DEEP_LOG
    jit_deep_log(function);
#endif
  }



  template<class T>
  void function_zero_rep_subtype_exec(JitFunction& function, OSubLattice<T>& dest, const Subset& s )
  {
    if (s.numSiteTable() < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());

    int th_count = s.numSiteTable();

    WorkgroupGuardExec workgroupGuardExec(th_count);
  
    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }


  
} // QDP
#endif
