#ifndef QDP_JITF_COPYMASK_H
#define QDP_JITF_COPYMASK_H


namespace QDP {

  template<class T,class T1>
  void
  function_copymask_build( JitFunction& function, OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
  {
    llvm_start_new_function("copymask",__PRETTY_FUNCTION__ );

    WorkgroupGuard workgroupGuard;

    ParamLeafScalar param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  typeJIT;
    typedef typename LeafFunctor<OLattice<T1>, ParamLeafScalar>::Type_t  maskJIT;

    typeJIT dest_jit(forEach(dest, param_leaf, TreeCombine()));
    typeJIT src_jit (forEach(src , param_leaf, TreeCombine()));
    maskJIT mask_jit(forEach(mask, param_leaf, TreeCombine()));

    llvm::Value* r_idx = llvm_thread_idx();

    workgroupGuard.check(r_idx);


    JitIf ifCopy( mask_jit.elemREG( JitDeviceLayout::Coalesced , r_idx ).elem().elem().elem().get_val() );
    {
      dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ) = src_jit.elemREG( JitDeviceLayout::Coalesced , r_idx );
    }
    ifCopy.end();
    

    jit_get_function(function);
  }



  template<class T,class T1>
  void 
  function_copymask_exec(JitFunction& function, OLattice<T>& dest, const OLattice<T1>& mask, const OLattice<T>& src )
  {
    AddressLeaf addr_leaf(all);

#ifdef QDP_DEEP_LOG
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    function.set_dest_id( dest.getId() );
    function.set_is_lat(true);
#endif

    forEach(dest, addr_leaf, NullCombine());
    forEach(src, addr_leaf, NullCombine());
    forEach(mask, addr_leaf, NullCombine());

    int th_count = Layout::sitesOnNode();

    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
    
    jit_launch( function , th_count , ids );

#ifdef QDP_DEEP_LOG
    jit_deep_log(function);
#endif
  }

}
#endif
