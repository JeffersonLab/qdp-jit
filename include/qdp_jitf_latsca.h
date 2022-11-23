#ifndef QDP_JITFUNC_LATSCA_H
#define QDP_JITFUNC_LATSCA_H


namespace QDP {


  template<class T, class T1, class Op, class RHS>
  void
  function_lat_sca_exec(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
    if (s.numSiteTable() < 1)
      return;

#ifdef QDP_DEEP_LOG
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    function.set_dest_id( dest.getId() );
    function.set_is_lat(true);
#endif


    int th_count = MasterMap::Instance().getCountInnerScalar(s,0);
    WorkgroupGuardExec workgroupGuardExec(th_count);

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());
    AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
    forEach(rhs, addr_leaf, NullCombine());
    
    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( MasterMap::Instance().getIdInnerScalar(s,0) );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);

#ifdef QDP_DEEP_LOG
    jit_deep_log(function);
#endif
  }


  template<class T, class T1, class Op, class RHS>
  void
  function_lat_sca_build(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
  {
    llvm_start_new_function("eval_lat_sca" , __PRETTY_FUNCTION__ );

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

    auto op_jit = AddOpParam<Op,ParamLeafScalar>::apply(op,param_leaf);

    typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeafScalar, TreeCombine>::Type_t View_t;
    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);
	    
    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );
	
    op_jit(dest_jit.elem( JitDeviceLayout::Coalesced , r_idx),
	   forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar , r_idx ), OpCombine()));
    
    jit_get_function( function );
  }


  template<class T, class T1, class Op, class RHS>
  void
  function_lat_sca_subtype_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
  {
    llvm_start_new_function("eval_lat_sca_subtype",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();
      
    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
    typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

    op_jit(dest_jit.elemScalar( JitDeviceLayout::Scalar , r_idx ),
	   forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar , r_idx ), OpCombine()));

    jit_get_function( function );
  }



  
  template<class T, class T1, class Op, class RHS>
  void 
  function_lat_sca_subtype_exec(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    WorkgroupGuardExec workgroupGuardExec(th_count);

    AddressLeaf addr_leaf(s);

    forEach(dest, addr_leaf, NullCombine());
    AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
    forEach(rhs, addr_leaf, NullCombine());

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( MasterMap::Instance().getIdInnerScalar(s,0) );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }


  
} // QDP
#endif
