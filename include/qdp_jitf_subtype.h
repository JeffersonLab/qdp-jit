#ifndef QDP_JITFUNC_SUBTYPE_H
#define QDP_JITFUNC_SUBTYPE_H


namespace QDP {

  
  template<class T, class C1, class Op, class RHS>
  void
  function_subtype_type_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs)
  {
    llvm_start_new_function("eval_subtype_type",__PRETTY_FUNCTION__ );

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeafScalar>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
    auto op_jit = AddOpParam<Op,ParamLeafScalar>::apply(op,param_leaf);
    typename ForEach<QDPExpr<RHS,C1 >, ParamLeafScalar, TreeCombine>::Type_t rhs_jit(forEach(rhs, param_leaf, TreeCombine()));

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

    op_jit( dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ),
	    forEach(rhs_jit, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));

    jit_get_function( function );
  }


  
  template<class T, class T1, class Op>
  void
  operator_type_subtype_build(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
  {
    typedef typename ScalarType< typename QDPSubType<T1,OLattice<T1>>::Subtype_t >::Type_t RT;
      
    llvm_start_new_function("eval_type_subtype",__PRETTY_FUNCTION__ );

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
    auto op_jit = AddOpParam<Op,ParamLeafScalar>::apply(op,param_leaf);
    typename LeafFunctor<QDPSubType<T1,OLattice<T1> >, ParamLeafScalar>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

    typename REGType< typename JITType< RT >::Type_t >::Type_t rhs_reg;
    rhs_reg.setup( rhs_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );
  
    op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), rhs_reg );

    jit_get_function( function );
  }


  
  template<class T, class T1, class Op>
  void
  operator_subtype_subtype_build(JitFunction& function, OSubLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
  {
    typedef typename ScalarType< typename QDPSubType<T1,OLattice<T1>>::Subtype_t >::Type_t RT;
      
    llvm_start_new_function("eval_subtype_subtype",__PRETTY_FUNCTION__ );

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
    typename LeafFunctor<QDPSubType<T1,OLattice<T1> >, ParamLeaf>::Type_t   rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
    
    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

    typename REGType< typename JITType< RT >::Type_t >::Type_t rhs_reg;
    rhs_reg.setup( rhs_jit.elemScalar( JitDeviceLayout::Scalar , r_idx_thread ) );
  
    op_jit( dest_jit.elemScalar( JitDeviceLayout::Scalar , r_idx_thread ), rhs_reg );

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

    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
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

    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
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

