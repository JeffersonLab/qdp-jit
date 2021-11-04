#ifndef QDP_JITFUNC_SUBTYPE_H
#define QDP_JITFUNC_SUBTYPE_H


namespace QDP {

  
  template<class T, class C1, class Op, class RHS>
  void
  function_subtype_type_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs)
  {
    llvm_start_new_function("eval_subtype_type",__PRETTY_FUNCTION__ );

    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
    typename ForEach<QDPExpr<RHS,C1 >, ParamLeaf, TreeCombine>::Type_t rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
  
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

    op_jit( dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ), // Coalesced
	    forEach(rhs_jit, ViewLeaf( JitDeviceLayout::Coalesced , r_idx_perm ), OpCombine()));

    jit_get_function( function );
  }


  
  template<class T, class T1, class Op>
  void
  operator_type_subtype_build(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
  {
    typedef typename QDPSubType<T1,OLattice<T1>>::Subtype_t RT;
      
    llvm_start_new_function("eval_type_subtype",__PRETTY_FUNCTION__ );

    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

    ParamLeaf param_leaf;

    typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
    //typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
    typename LeafFunctor<QDPSubType<T1,OLattice<T1> >, ParamLeaf>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
    //typename ForEach<QDPExpr<RHS,OSubLattice<T1> >, ParamLeaf, TreeCombine>::Type_t rhs_jit(forEach(rhs, param_leaf, TreeCombine()));

    
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType< RT >::Type_t >::Type_t rhs_reg;
    rhs_reg.setup( rhs_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );
  
    // op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx_thread ), // Coalesced
    // 	  forEach(rhs_jit, ViewLeaf( JitDeviceLayout::Scalar , r_idx_perm ), OpCombine()));

    op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx_perm ), rhs_reg );

    jit_get_function( function );
  }


  
  template<class T, class T1, class Op>
  void
  operator_subtype_subtype_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
  {
    typedef typename QDPSubType<T1,OLattice<T1>>::Subtype_t RT;
      
    llvm_start_new_function("eval_subtype_subtype",__PRETTY_FUNCTION__ );

    ParamRef p_th_count     = llvm_add_param<int>();
    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
    typename LeafFunctor<QDPSubType<T1,OLattice<T1> >, ParamLeaf>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
    
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    typename REGType< typename JITType< RT >::Type_t >::Type_t rhs_reg;
    rhs_reg.setup( rhs_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );
  
    op_jit( dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ), rhs_reg );

    jit_get_function( function );
  }


  template<class T, class C1, class Op, class RHS>
  void 
  function_subtype_type_exec(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs, const Subset& s)
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI node\n";
      return;
    }

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());
    AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
    forEach(rhs, addr_leaf, NullCombine());

    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }



  template<class T, class T1, class Op>
  void 
  operator_type_subtype_exec(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs, const Subset& s)
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());
    AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
    forEach(rhs, addr_leaf, NullCombine());

    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }



  template<class T, class T1, class Op>
  void 
  operator_subtype_subtype_exec(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs, const Subset& s)
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());
    AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
    forEach(rhs, addr_leaf, NullCombine());

    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }


} // QDP
#endif

