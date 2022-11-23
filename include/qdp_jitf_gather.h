#ifndef QDP_JITFUNC_GATHER_H
#define QDP_JITFUNC_GATHER_H

namespace QDP {


  template<class T, class T1, class RHS>
#if defined (QDP_PROP_OPT)
  typename std::enable_if_t< ! HasProp<RHS>::value >
#else
  void
#endif  
  function_gather_build( JitFunction& function, const QDPExpr<RHS,OLattice<T1> >& rhs )
  {
    typedef typename WordType<T1>::Type_t WT;

    llvm_start_new_function("gather",__PRETTY_FUNCTION__);
    
    WorkgroupGuard workgroupGuard;

    ParamRef p_soffset = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(OLattice<T>(), param_leaf, TreeCombine()));

    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeafScalar, TreeCombine>::Type_t View_t;
    View_t rhs_view( forEach( rhs , param_leaf , TreeCombine() ) );


    llvm::Value * r_idx     = llvm_thread_idx();

    workgroupGuard.check(r_idx);

    llvm::Value * r_idx_site = llvm_array_type_indirection<int>( p_soffset , r_idx );
  
    OpAssign()( dest_jit.elem( JitDeviceLayout::Scalar , r_idx ) , 
		forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx_site ) , OpCombine() ) );

    jit_get_function( function );
  }


#if defined (QDP_PROP_OPT)
  template<class T, class T1, class RHS>
  typename std::enable_if_t< HasProp<RHS>::value >
  function_gather_build( JitFunction& function, const QDPExpr<RHS,OLattice<T1> >& rhs )
  {
    typedef typename WordType<T1>::Type_t WT;

    llvm_start_new_function("gather_prop",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;

    ParamRef p_soffset = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeafScalar>::Type_t  FuncRet_t;
    FuncRet_t dest_jit(forEach(OLattice<T>(), param_leaf, TreeCombine()));

    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeafScalar, TreeCombine>::Type_t View_t;
    View_t rhs_view( forEach( rhs , param_leaf , TreeCombine() ) );

    llvm::Value * r_idx     = llvm_thread_idx();  

    workgroupGuard.check(r_idx);

    llvm::Value * r_idx_site = llvm_array_type_indirection<int>( p_soffset , r_idx );

  
    std::vector< JitForLoop > loops;
    CreateLoops<T,OpAssign>::apply( loops , OpAssign() );

	  
    ViewSpinLeaf viewSpin( JitDeviceLayout::Coalesced , r_idx_site );
    for( int i = 0 ; i < loops.size() ; ++i )
      viewSpin.indices.push_back( loops.at(i).index() );

    ViewSpinLeaf viewSpinDest( JitDeviceLayout::Scalar , r_idx );
    for( int i = 0 ; i < loops.size() ; ++i )
      viewSpinDest.indices.push_back( loops.at(i).index() );

    
    OpAssign()( viewSpinJit( dest_jit , viewSpinDest ) , forEach( rhs_view , viewSpin , OpCombine() ) );
 
    for( int i = loops.size() - 1 ; 0 <= i ; --i )
      {
	loops[i].end();
      }

    jit_get_function( function );
  }
#endif


  template<class T1, class RHS>
  void
  function_gather_exec( JitFunction& function, int send_buf_id , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs , const Subset& subset )
  {
    if (subset.numSiteTable() < 1)
      return;

    AddressLeaf addr_leaf(subset);

    forEach(rhs, addr_leaf, NullCombine());

    int th_count = map.soffset(subset).size();

    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( map.getSoffsetsId(subset) );
    ids.push_back( send_buf_id );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,th_count,ids);

#ifdef QDP_DEEP_LOG
    jit_deep_log(function);
#endif
  }

} // QDP
#endif
