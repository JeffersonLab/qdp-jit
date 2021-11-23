#ifndef QDP_JITFUNC_POKESITE_H
#define QDP_JITFUNC_POKESITE_H


namespace QDP {


  template<class T, class T1>
  void
  function_pokeSite_build( JitFunction& function, const OLattice<T>& dest , const OScalar<T1>& r  )
  {
    llvm_start_new_function("eval_pokeSite",__PRETTY_FUNCTION__);

    ParamRef p_siteindex    = llvm_add_param<int>();

    ParamLeafScalar param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    
    auto op_jit = AddOpParam<OpAssign,ParamLeafScalar>::apply(OpAssign(),param_leaf);

    typedef typename LeafFunctor<OScalar<T1>, ParamLeafScalar>::Type_t  FuncRet_t1;
    FuncRet_t1 r_jit(forEach(r, param_leaf, TreeCombine()));

    llvm::Value* r_siteindex = llvm_derefParam( p_siteindex );
    llvm::Value* r_zero      = llvm_create_value(0);

    op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_siteindex ), 
	    LeafFunctor< FuncRet_t1 , ViewLeaf >::apply( r_jit , ViewLeaf( JitDeviceLayout::Scalar , r_zero ) ) );

    jit_get_function( function );
  }


  template<class T, class T1>
  void 
  function_pokeSite_exec(JitFunction& function, OLattice<T>& dest, const OScalar<T1>& rhs, const multi1d<int>& coord )
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    AddressLeaf addr_leaf(all);

    forEach(dest, addr_leaf, NullCombine());
    forEach(rhs, addr_leaf, NullCombine());

    JitParam jit_siteindex( QDP_get_global_cache().addJitParamInt( Layout::linearSiteIndex(coord) ) );

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_siteindex.get_id() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,1,ids);   // 1 - thread count
  }

} // QDP
#endif
