#ifndef QDP_JITFUNC_OP_H
#define QDP_JITFUNC_OP_H


namespace QDP {


  template<class OP,class T, class QT1, class QT2>
  void 
  function_OP_exec(JitFunction& function, OSubLattice<T>& ret,
		   const QT1& l,const QT2& r,
		   const Subset& s)
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    AddressLeaf addr_leaf(s);
    OP op;
    forEach(ret, addr_leaf, NullCombine());
    AddOpAddress<OP,AddressLeaf>::apply(op,addr_leaf);
    forEach(l, addr_leaf, NullCombine());
    forEach(r, addr_leaf, NullCombine());

    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_th_count.get_id() );
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);
  }



  
  template<class OP,class T, class T1, class T2, class C1, class C2>
  void
  function_OP_type_subtype_build(JitFunction& function, OSubLattice<T>& ret,
				 const QDPType<T1,C1> & l,const QDPSubType<T2,C2> & r)
  {
    typedef typename QDPType<T1,C1>::Subtype_t    LT;
    typedef typename QDPSubType<T2,C2>::Subtype_t RT;
    
    llvm_start_new_function("localInnerProduct_type_subtype",__PRETTY_FUNCTION__ );

    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   ret_jit(forEach(ret, param_leaf, TreeCombine()));

    OP op;
    auto op_jit = AddOpParam<OP,ParamLeaf>::apply(op,param_leaf);

    typename LeafFunctor<QDPType<T1,C1>   , ParamLeaf>::Type_t   l_jit(forEach(l, param_leaf, TreeCombine()));
    typename LeafFunctor<QDPSubType<T2,C2>, ParamLeaf>::Type_t   r_jit(forEach(r, param_leaf, TreeCombine()));
	
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType< LT >::Type_t >::Type_t l_reg;
    l_reg.setup( l_jit.elem( JitDeviceLayout::Coalesced , r_idx_perm ) );

    typename REGType< typename JITType< RT >::Type_t >::Type_t r_reg;
    r_reg.setup( r_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );

    ret_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) = op_jit( l_reg , r_reg );
    
    jit_get_function(function);
  }

  
  template<class OP, class T, class T1, class T2, class C1, class C2>
  void
  function_OP_subtype_type_build(JitFunction& function, OSubLattice<T>& ret,
				 const QDPSubType<T1,C1> & l,const QDPType<T2,C2> & r)
  {
    typedef typename QDPSubType<T1,C1>::Subtype_t LT;
    typedef typename QDPType<T2,C2>::Subtype_t    RT;
    
    llvm_start_new_function("localInnerProduct_subtype_type",__PRETTY_FUNCTION__ );

    
    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   ret_jit(forEach(ret, param_leaf, TreeCombine()));

    OP op;
    auto op_jit = AddOpParam<OP,ParamLeaf>::apply(op,param_leaf);

    typename LeafFunctor<QDPSubType<T1,C1> , ParamLeaf>::Type_t   l_jit(forEach(l, param_leaf, TreeCombine()));
    typename LeafFunctor<QDPType<T2,C2>    , ParamLeaf>::Type_t   r_jit(forEach(r, param_leaf, TreeCombine()));
	
    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
    llvm::Value* r_idx_thread = llvm_thread_idx();

    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

    typename REGType< typename JITType< LT >::Type_t >::Type_t l_reg;
    l_reg.setup( l_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) );   // ok: Scalar

    typename REGType< typename JITType< RT >::Type_t >::Type_t r_reg;
    r_reg.setup( r_jit.elem( JitDeviceLayout::Coalesced , r_idx_perm ) );

    ret_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ) = op_jit( l_reg , r_reg );   // ok: Scalar
    
    jit_get_function( function );
  }

} // QDP
#endif