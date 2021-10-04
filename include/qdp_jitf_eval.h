#ifndef QDP_JITFUNC_EVAL_H
#define QDP_JITFUNC_EVAL_H

#include<type_traits>

namespace QDP {

  template<class T, class T1, class Op, class RHS>
#if defined (QDP_PROP_OPT)
  typename std::enable_if_t< ! HasProp<RHS>::value >
#else
  void
#endif  
  function_build(JitFunction& function, const DynKey& key, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
    std::ostringstream expr;
#if 0
    printExprTreeSubset( expr , dest, op, rhs , s , key );
#else
    expr << std::string(__PRETTY_FUNCTION__) << "_key=" << key;
#endif
  
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , expr.str().c_str() );
	if (!function.empty())
	  return;
      }
    llvm_start_new_function("eval",expr.str().c_str() );
  
    if ( key.get_offnode_comms() )
      {
	if ( s.hasOrderedRep() )
	  {
	    ParamRef p_th_count   = llvm_add_param<int>();
	    ParamRef p_site_table = llvm_add_param<int*>();

	    ParamLeaf param_leaf;

	    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	    llvm::Value* r_idx_thread = llvm_thread_idx();
       
	    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	    op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		    forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));	  
	  }
	else
	  {
	    QDPIO::cout << "eval with shifts on unordered subsets not supported" << std::endl;
	    QDP_abort(1);
	  }
      }
    else
      {
	if ( s.hasOrderedRep() )
	  {
	    ParamRef p_th_count = llvm_add_param<int>();
	    ParamRef p_start    = llvm_add_param<int>();

	    ParamLeaf param_leaf;

	    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
	    llvm::Value * r_start        = llvm_derefParam( p_start );

	    llvm::Value* r_idx_thread = llvm_thread_idx();

	    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	    llvm::Value* r_idx = llvm_add( r_idx_thread , r_start );

	    op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		    forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));
	  }
	else // unordered Subset
	  {
	    ParamRef p_th_count   = llvm_add_param<int>();
	    ParamRef p_site_table = llvm_add_param<int*>();

	    ParamLeaf param_leaf;

	    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	    llvm::Value* r_idx_thread = llvm_thread_idx();
       
	    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	    op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		    forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));	  
	  }
      }
  
    jit_get_function( function );
  }


  
#if defined (QDP_PROP_OPT)
  template<class T, class T1, class Op, class RHS>
  typename std::enable_if_t< HasProp<RHS>::value >
  function_build(JitFunction& function, const DynKey& key, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
    std::ostringstream expr;

    //expr << std::string(__PRETTY_FUNCTION__) << "_key=" << key;
  
    if (ptx_db::db_enabled)
      {
	llvm_ptx_db( function , expr.str().c_str() );
	if (!function.empty())
	  return;
      }
    llvm_start_new_function("evalp",expr.str().c_str() );
  
    if ( key.get_offnode_comms() )
      {
	if ( s.hasOrderedRep() )
	  {
	    ParamRef p_th_count   = llvm_add_param<int>();
	    ParamRef p_site_table = llvm_add_param<int*>();

	    ParamLeaf param_leaf;

	    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

	    typedef typename AddOpParam<Op,ParamLeaf>::Type_t OpJit_t;
	    OpJit_t op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	    llvm::Value* r_idx_thread = llvm_thread_idx();
       
	    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	    std::vector< JitForLoop > loops;
	    CreateLoops<T,OpJit_t>::apply( loops , op_jit );
	  
	    ViewSpinLeaf viewSpin( JitDeviceLayout::Coalesced , r_idx );

	    for( int i = 0 ; i < loops.size() ; ++i )
	      viewSpin.indices.push_back( loops.at(i).index() );

	    op_jit( viewSpinJit( dest_jit , viewSpin ) , forEach( rhs_view , viewSpin , OpCombine() ) );
 
	    for( int i = loops.size() - 1 ; 0 <= i ; --i )
	      {
		loops[i].end();
	      }
	  
	  }
	else
	  {
	    QDPIO::cout << "eval with shifts on unordered subsets not supported" << std::endl;
	    QDP_abort(1);
	  }
      }
    else
      {
	if ( s.hasOrderedRep() )
	  {
	    ParamRef p_th_count = llvm_add_param<int>();
	    ParamRef p_start    = llvm_add_param<int>();

	    ParamLeaf param_leaf;

	    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

	    typedef typename AddOpParam<Op,ParamLeaf>::Type_t OpJit_t;
	    OpJit_t op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
	    llvm::Value * r_start        = llvm_derefParam( p_start );

	    llvm::Value* r_idx_thread = llvm_thread_idx();

	    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	    llvm::Value* r_idx = llvm_add( r_idx_thread , r_start );

	    std::vector< JitForLoop > loops;
	    CreateLoops<T,OpJit_t>::apply( loops , op_jit );
	  
	    ViewSpinLeaf viewSpin( JitDeviceLayout::Coalesced , r_idx );

	    for( int i = 0 ; i < loops.size() ; ++i )
	      viewSpin.indices.push_back( loops.at(i).index() );

	    op_jit( viewSpinJit( dest_jit , viewSpin ) , forEach( rhs_view , viewSpin , OpCombine() ) );
 
	    for( int i = loops.size() - 1 ; 0 <= i ; --i )
	      {
		loops[i].end();
	      }
	  }
	else // unordered Subset
	  {
	    ParamRef p_th_count   = llvm_add_param<int>();
	    ParamRef p_site_table = llvm_add_param<int*>();

	    ParamLeaf param_leaf;

	    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
	    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
	  
	    auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

	    typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
	    View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

	    llvm::Value * r_th_count     = llvm_derefParam( p_th_count );

	    llvm::Value* r_idx_thread = llvm_thread_idx();
       
	    llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

	    llvm::Value* r_idx = llvm_array_type_indirection( p_site_table , r_idx_thread );

	    op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
		    forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));	  
	  }
      }
  
    jit_get_function( function );
  }
#endif
  

  template<class T, class T1, class Op, class RHS>
  void 
  function_exec(JitFunction& function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.start = s.start();
    function.count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
    function.size_T = sizeof(T);
    function.type_W = typeid(typename WordType<T>::Type_t).name();
    function.set_dest_id( dest.getId() );
#endif
    
    if (s.numSiteTable() < 1)
      return;

    ShiftPhase1 phase1(s);
    int offnode_maps = forEach(rhs, phase1 , BitOrCombine());

    AddressLeaf addr_leaf(s);
    forEach(dest, addr_leaf, NullCombine());
    AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
    forEach(rhs, addr_leaf, NullCombine());

    // For tuning
    function.set_dest_id( dest.getId() );
    function.set_enable_tuning();
  
    if (offnode_maps == 0)
      {
	if ( s.hasOrderedRep() )
	  {
	    int th_count = s.numSiteTable();
		
	    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
	    JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );

	    std::vector<QDPCache::ArgKey> ids;
	    ids.push_back( jit_th_count.get_id() );
	    ids.push_back( jit_start.get_id() );
	    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	      ids.push_back( addr_leaf.ids[i] );
	  
	    jit_launch(function,th_count,ids);
	  }
	else
	  {
	    int th_count = s.numSiteTable();
      
	    JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

	    std::vector<QDPCache::ArgKey> ids;
	    ids.push_back( jit_th_count.get_id() );
	    ids.push_back( s.getIdSiteTable() );
	    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
	      ids.push_back( addr_leaf.ids[i] );
 
	    jit_launch(function,th_count,ids);
	  }
      }
    else
      {
	// 1st. call: inner
	{
	  int th_count = MasterMap::Instance().getCountInner(s,offnode_maps);
      
	  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

	  std::vector<QDPCache::ArgKey> ids;
	  ids.push_back( jit_th_count.get_id() );
	  ids.push_back( MasterMap::Instance().getIdInner(s,offnode_maps) );
	  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	    ids.push_back( addr_leaf.ids[i] );
 
	  jit_launch(function,th_count,ids);
	}
      
	// 2nd call: face
	{
	  ShiftPhase2 phase2;
	  forEach(rhs, phase2 , NullCombine());

	  int th_count = MasterMap::Instance().getCountFace(s,offnode_maps);
      
	  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );

	  std::vector<QDPCache::ArgKey> ids;
	  ids.push_back( jit_th_count.get_id() );
	  ids.push_back( MasterMap::Instance().getIdFace(s,offnode_maps) );
	  for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	    ids.push_back( addr_leaf.ids[i] );
 
	  jit_launch(function,th_count,ids);
	}
      
      }
  }  

} // QDP
#endif
