#ifndef QDP_JITFUNC_GATHER_H
#define QDP_JITFUNC_GATHER_H

namespace QDP {


  template<class T, class T1, class RHS>
  typename std::enable_if_t< ! HasProp<RHS>::value >
  function_gather_build( JitFunction& function, const QDPExpr<RHS,OLattice<T1> >& rhs )
  {
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , __PRETTY_FUNCTION__ );
	if (!function.empty())
	  return;
      }


    typedef typename WordType<T1>::Type_t WT;

    llvm_start_new_function("gather",__PRETTY_FUNCTION__);

    ParamRef p_lo      = llvm_add_param<int>();
    ParamRef p_hi      = llvm_add_param<int>();
    ParamRef p_soffset = llvm_add_param<int*>();
    ParamRef p_sndbuf  = llvm_add_param<WT*>();

    ParamLeaf param_leaf;

    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
    View_t rhs_view( forEach( rhs , param_leaf , TreeCombine() ) );

    typedef typename JITType< OLattice<T> >::Type_t DestView_t;
    DestView_t dest_jit( p_sndbuf );

    llvm_derefParam( p_lo );  // r_lo not used
    llvm::Value * r_hi      = llvm_derefParam( p_hi );
    llvm::Value * r_idx     = llvm_thread_idx();  

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    llvm::Value * r_idx_site = llvm_array_type_indirection( p_soffset , r_idx );
  
    OpAssign()( dest_jit.elem( JitDeviceLayout::Scalar , r_idx ) , 
		forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx_site ) , OpCombine() ) );

    jit_get_function( function );
  }


  template<class T, class T1, class RHS>
  typename std::enable_if_t< HasProp<RHS>::value >
  function_gather_build( JitFunction& function, const QDPExpr<RHS,OLattice<T1> >& rhs )
  {
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , __PRETTY_FUNCTION__ );
	if (!function.empty())
	  return;
      }


    typedef typename WordType<T1>::Type_t WT;

    llvm_start_new_function("gather_prop",__PRETTY_FUNCTION__);

    ParamRef p_lo      = llvm_add_param<int>();
    ParamRef p_hi      = llvm_add_param<int>();
    ParamRef p_soffset = llvm_add_param<int*>();
    ParamRef p_sndbuf  = llvm_add_param<WT*>();

    ParamLeaf param_leaf;

    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
    View_t rhs_view( forEach( rhs , param_leaf , TreeCombine() ) );

    typedef typename JITType< OLattice<T> >::Type_t DestView_t;
    DestView_t dest_jit( p_sndbuf );

    llvm_derefParam( p_lo );  // r_lo not used
    llvm::Value * r_hi      = llvm_derefParam( p_hi );
    llvm::Value * r_idx     = llvm_thread_idx();  

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    llvm::Value * r_idx_site = llvm_array_type_indirection( p_soffset , r_idx );

  
    std::vector< JitForLoop > loops;
    CreateLoops<T,OpAssign>::apply( loops , OpAssign() );
    //create_dest_loops<T>( loops ,  );
	  
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



  template<class T1, class RHS>
  void
  function_gather_exec( JitFunction& function, int send_buf_id , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs , const Subset& subset )
  {
    if (subset.numSiteTable() < 1)
      return;

    AddressLeaf addr_leaf(subset);

    forEach(rhs, addr_leaf, NullCombine());

    int hi = map.soffset(subset).size();

    JitParam jit_lo( QDP_get_global_cache().addJitParamInt( 0 ) );        // lo, leave it in
    JitParam jit_hi( QDP_get_global_cache().addJitParamInt( hi ) );

    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_lo.get_id() );
    ids.push_back( jit_hi.get_id() );
    ids.push_back( map.getSoffsetsId(subset) );
    ids.push_back( send_buf_id );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
 
    jit_launch(function,hi,ids);
  }

} // QDP
#endif
