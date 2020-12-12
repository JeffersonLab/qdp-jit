#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H



namespace QDP {



  template<class OP,class T, class QT1, class QT2>
  void 
  function_OP_exec(JitFunction function, OSubLattice<T>& ret,
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
  JitFunction
  function_OP_type_subtype_build(OSubLattice<T>& ret,
				 const QDPType<T1,C1> & l,const QDPSubType<T2,C2> & r)
  {
    typedef typename QDPType<T1,C1>::Subtype_t    LT;
    typedef typename QDPSubType<T2,C2>::Subtype_t RT;
    
    if (ptx_db::db_enabled) {
      JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (!func.empty())
	return func;
    }

    llvm_start_new_function();

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
    
    return jit_function_epilogue_get_cuf("jit_localInnerProduct_type_subtype.ptx" , __PRETTY_FUNCTION__ );
  }

  
  template<class OP, class T, class T1, class T2, class C1, class C2>
  JitFunction
  function_OP_subtype_type_build(OSubLattice<T>& ret,
				 const QDPSubType<T1,C1> & l,const QDPType<T2,C2> & r)
  {
    typedef typename QDPSubType<T1,C1>::Subtype_t LT;
    typedef typename QDPType<T2,C2>::Subtype_t    RT;
    
    if (ptx_db::db_enabled) {
      JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (!func.empty())
	return func;
    }

    llvm_start_new_function();

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
    
    return jit_function_epilogue_get_cuf("jit_localInnerProduct_type_subtype.ptx" , __PRETTY_FUNCTION__ );
  }





  
template<class T, class T1, class Op, class RHS>
JitFunction
function_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

  ParamRef p_ordered      = llvm_add_param<bool>();
  ParamRef p_th_count     = llvm_add_param<int>();
  ParamRef p_start        = llvm_add_param<int>();
  ParamRef p_end          = llvm_add_param<int>();
  ParamRef p_do_site_perm = llvm_add_param<bool>();
  ParamRef p_site_table   = llvm_add_param<int*>();
  ParamRef p_member_array = llvm_add_param<bool*>();
  

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));


  llvm::Value * r_ordered      = llvm_derefParam( p_ordered );
  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
   llvm::Value * r_start        = llvm_derefParam( p_start );
   llvm::Value * r_end          = llvm_derefParam( p_end );
   llvm::Value * r_do_site_perm = llvm_derefParam( p_do_site_perm );

  llvm::Value* r_no_site_perm = llvm_not( r_do_site_perm );  
  //mainFunc->dump();
  llvm::Value* r_idx_thread = llvm_thread_idx();
  //mainFunc->dump();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  llvm::BasicBlock * block_no_site_perm_exit = llvm_new_basic_block();
  llvm::BasicBlock * block_no_site_perm = llvm_new_basic_block();
  llvm::BasicBlock * block_site_perm = llvm_new_basic_block();
  llvm::BasicBlock * block_add_start = llvm_new_basic_block();
  llvm::BasicBlock * block_add_start_else = llvm_new_basic_block();

  llvm::Value* r_idx_perm_phi0;
  llvm::Value* r_idx_perm_phi1;

  llvm_cond_branch( r_no_site_perm , block_no_site_perm , block_site_perm ); 
  {
    llvm_set_insert_point(block_site_perm);
    r_idx_perm_phi0 = llvm_array_type_indirection( p_site_table , r_idx_thread ); // PHI 0   
    llvm_branch( block_no_site_perm_exit );
  }
  {
    llvm_set_insert_point(block_no_site_perm);
    llvm_cond_branch( r_ordered , block_add_start , block_add_start_else );
    {
      llvm_set_insert_point(block_add_start);
      r_idx_perm_phi1 = llvm_add( r_idx_thread , r_start ); // PHI 1  
      llvm_branch( block_no_site_perm_exit );
      llvm_set_insert_point(block_add_start_else);
      llvm_branch( block_no_site_perm_exit );
    }
  }
  llvm_set_insert_point(block_no_site_perm_exit);

  llvm::PHINode* r_idx = llvm_phi( r_idx_perm_phi0->getType() , 3 );

  r_idx->addIncoming( r_idx_perm_phi0 , block_site_perm );
  r_idx->addIncoming( r_idx_perm_phi1 , block_add_start );
  r_idx->addIncoming( r_idx_thread , block_add_start_else );

  llvm::BasicBlock * block_ordered = llvm_new_basic_block();
  llvm::BasicBlock * block_not_ordered = llvm_new_basic_block();
  llvm::BasicBlock * block_ordered_exit = llvm_new_basic_block();
  llvm_cond_branch( r_ordered , block_ordered , block_not_ordered );
  {
    llvm_set_insert_point(block_not_ordered);
    llvm::Value* r_ismember     = llvm_array_type_indirection( p_member_array , r_idx );
    llvm::Value* r_ismember_not = llvm_not( r_ismember );
    llvm_cond_exit( r_ismember_not ); 
    llvm_branch( block_ordered_exit );
  }
  {
    llvm_set_insert_point(block_ordered);
    llvm_cond_exit( llvm_gt( r_idx , r_end ) );
    llvm_cond_exit( llvm_lt( r_idx , r_start ) );
    llvm_branch( block_ordered_exit );
  }
  llvm_set_insert_point(block_ordered_exit);



  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ), 
	  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , r_idx ), OpCombine()));


  return jit_function_epilogue_get_cuf("jit_eval.ptx" , __PRETTY_FUNCTION__ );
}




template<class T, class C1, class Op, class RHS>
JitFunction
function_subtype_type_build(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs)
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

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

  return jit_function_epilogue_get_cuf("jit_eval_subtype_expr.ptx" , __PRETTY_FUNCTION__ );
}


  
template<class T, class T1, class Op>
JitFunction
operator_type_subtype_build(OLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
{
  typedef typename QDPSubType<T1,OLattice<T1>>::Subtype_t RT;
      
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

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

  return jit_function_epilogue_get_cuf("jit_eval_subtype_expr.ptx" , __PRETTY_FUNCTION__ );
}


  
template<class T, class T1, class Op>
JitFunction
operator_subtype_subtype_build(OSubLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs)
{
  typedef typename QDPSubType<T1,OLattice<T1>>::Subtype_t RT;
      
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

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

  return jit_function_epilogue_get_cuf("jit_eval_subtype_expr.ptx" , __PRETTY_FUNCTION__ );
}


  

template<class T, class T1, class Op, class RHS>
void
function_lat_sca_exec(JitFunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  if (s.numSiteTable() < 1)
    return;

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  int th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();

  JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
  JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_ordered.get_id() );
  ids.push_back( jit_th_count.get_id() );
  ids.push_back( jit_start.get_id() );
  ids.push_back( jit_end.get_id() );
  ids.push_back( s.getIdMemberTable() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



  


template<class T, class T1, class Op, class RHS>
JitFunction
function_lat_sca_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  std::vector<ParamRef> params = jit_function_preamble_param();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  llvm::Value * r_idx = jit_function_preamble_get_idx( params );

  op_jit(dest_jit.elem( JitDeviceLayout::Coalesced , r_idx), 
	 forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar , r_idx ), OpCombine()));

  return jit_function_epilogue_get_cuf("jit_lat_sca.ptx" , __PRETTY_FUNCTION__ );
}






template<class T, class T1, class Op, class RHS>
JitFunction
function_lat_sca_subtype_build(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

  ParamRef p_th_count     = llvm_add_param<int>();
      
  ParamLeaf param_leaf;

  typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
  typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  llvm::Value * r_th_count  = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  op_jit(dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ), // Coalesced
	 forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar , r_idx_thread ), OpCombine()));

  return jit_function_epilogue_get_cuf("jit_lat_sca.ptx" , __PRETTY_FUNCTION__ );
}

  



template<class T, class T1>
JitFunction
function_pokeSite_build( const OLattice<T>& dest , const OScalar<T1>& r  )
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

  ParamRef p_siteindex    = llvm_add_param<int>();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    
  auto op_jit = AddOpParam<OpAssign,ParamLeaf>::apply(OpAssign(),param_leaf);

  typedef typename LeafFunctor<OScalar<T1>, ParamLeaf>::Type_t  FuncRet_t1;
  FuncRet_t1 r_jit(forEach(r, param_leaf, TreeCombine()));

  llvm::Value* r_siteindex = llvm_derefParam( p_siteindex );
  llvm::Value* r_zero      = llvm_create_value(0);

  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , r_siteindex ), 
	  LeafFunctor< FuncRet_t1 , ViewLeaf >::apply( r_jit , ViewLeaf( JitDeviceLayout::Scalar , r_zero ) ) );

  return jit_function_epilogue_get_cuf("jit_pokesite.ptx" , __PRETTY_FUNCTION__ );
}


template<class T, class T1>
void 
function_pokeSite_exec(JitFunction function, OLattice<T>& dest, const OScalar<T1>& rhs, const multi1d<int>& coord )
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





template<class T>
JitFunction
function_zero_rep_build(OLattice<T>& dest)
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  std::vector<ParamRef> params = jit_function_preamble_param();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  llvm::Value * r_idx = jit_function_preamble_get_idx( params );

  zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

  return jit_function_epilogue_get_cuf("jit_zero.ptx" , __PRETTY_FUNCTION__ );
}



template<class T>
JitFunction
function_zero_rep_subtype_build(OSubLattice<T>& dest)
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

  ParamRef p_th_count     = llvm_add_param<int>();

  ParamLeaf param_leaf;
  typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  
  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx_thread) );

  return jit_function_epilogue_get_cuf("jit_zero_rep_subtype.ptx" , __PRETTY_FUNCTION__ );
}
  






template<class T>
JitFunction
function_random_build(OLattice<T>& dest , Seed& seed_tmp)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  llvm_start_new_function();

  // No member possible here.
  // If thread exits due to non-member
  // it possibly can't store the new seed at the end.

  ParamRef p_lo     = llvm_add_param<int>();
  ParamRef p_hi     = llvm_add_param<int>();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // RNG::ran_seed
  typedef typename LeafFunctor<Seed, ParamLeaf>::Type_t  SeedJIT;
  typedef typename LeafFunctor<LatticeSeed, ParamLeaf>::Type_t  LatticeSeedJIT;
  typedef typename REGType<typename SeedJIT::Subtype_t>::Type_t PSeedREG;

  SeedJIT ran_seed_jit(forEach(RNG::ran_seed, param_leaf, TreeCombine()));
  SeedJIT seed_tmp_jit(forEach(seed_tmp, param_leaf, TreeCombine()));
  SeedJIT ran_mult_n_jit(forEach(RNG::ran_mult_n, param_leaf, TreeCombine()));
  LatticeSeedJIT lattice_ran_mult_jit(forEach( *RNG::lattice_ran_mult , param_leaf, TreeCombine()));

  llvm::Value * r_lo     = llvm_derefParam( p_lo );
  llvm::Value * r_hi     = llvm_derefParam( p_hi );

  llvm::Value * r_idx_thread = llvm_thread_idx();

  llvm::Value * r_out_of_range       = llvm_gt( r_idx_thread , llvm_sub( r_hi , r_lo ) );
  llvm_cond_exit(  r_out_of_range );

  llvm::Value * r_idx = llvm_add( r_lo , r_idx_thread );

  PSeedREG seed_reg;
  PSeedREG skewed_seed_reg;
  PSeedREG ran_mult_n_reg;
  PSeedREG lattice_ran_mult_reg;

  seed_reg.setup( ran_seed_jit.elem() );

  lattice_ran_mult_reg.setup( lattice_ran_mult_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

  skewed_seed_reg = seed_reg * lattice_ran_mult_reg;

  ran_mult_n_reg.setup( ran_mult_n_jit.elem() );

  fill_random( dest_jit.elem(JitDeviceLayout::Coalesced,r_idx) , seed_reg , skewed_seed_reg , ran_mult_n_reg );

  llvm::Value * r_save = llvm_eq( r_idx_thread , llvm_create_value(0) );

  llvm::BasicBlock * block_save = llvm_new_basic_block();
  llvm::BasicBlock * block_not_save = llvm_new_basic_block();
  llvm::BasicBlock * block_save_exit = llvm_new_basic_block();
  llvm_cond_branch( r_save , block_save , block_not_save );
  {
    llvm_set_insert_point(block_save);
    seed_tmp_jit.elem() = seed_reg;
    llvm_branch( block_save_exit );
  }
  {
    llvm_set_insert_point(block_not_save);
    llvm_branch( block_save_exit );
  }
  llvm_set_insert_point(block_save_exit);

  return jit_function_epilogue_get_cuf("jit_random.ptx" , __PRETTY_FUNCTION__ );
}






template<class T, class T1, class RHS>
JitFunction
//function_gather_build( void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
function_gather_build( const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  if (ptx_db::db_enabled) {
    JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (!func.empty())
      return func;
  }

  typedef typename WordType<T1>::Type_t WT;

  llvm_start_new_function();

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

  return jit_function_epilogue_get_cuf("jit_gather.ll" , __PRETTY_FUNCTION__ );
}





template<class T1, class RHS>
void
  function_gather_exec( JitFunction function, int send_buf_id , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs , const Subset& subset )
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
#if 0  
  int lo = 0;
  int hi = map.soffset(subset).size();

  int soffsetsId = map.getSoffsetsId(subset);
  void * soffsetsDev = QDP_get_global_cache().getDevicePtr( soffsetsId );

  std::vector<void*> addr;

  addr.push_back( &lo );
  addr.push_back( &hi );
  addr.push_back( &soffsetsDev );
  addr.push_back( &send_buf );
  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
  }
  jit_launch(function,hi-lo,addr);
#endif
}


namespace COUNT {
  extern int count;
}


template<class T, class T1, class Op, class RHS>
void 
function_exec(JitFunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  //QDPIO::cout << __PRETTY_FUNCTION__ << "\n";

  if (s.numSiteTable() < 1)
    return;

  ShiftPhase1 phase1(s);
  int offnode_maps = forEach(rhs, phase1 , BitOrCombine());

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  if (offnode_maps == 0)
    {
      int th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();

      JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
      JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
      JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
      JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
      JitParam jit_do_soffset_index( QDP_get_global_cache().addJitParamBool( false ) );   // do soffset index

      std::vector<QDPCache::ArgKey> ids;
      ids.push_back( jit_ordered.get_id() );
      ids.push_back( jit_th_count.get_id() );
      ids.push_back( jit_start.get_id() );
      ids.push_back( jit_end.get_id() );
      ids.push_back( jit_do_soffset_index.get_id() );
      ids.push_back( -1 );  // soffset index table
      ids.push_back( s.getIdMemberTable() );
      for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	ids.push_back( addr_leaf.ids[i] );
 
      jit_launch(function,th_count,ids);
    }
  else
    {
      // 1st. call: inner
      {
	int th_count = MasterMap::Instance().getCountInner(s,offnode_maps);
      
	JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
	JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
	JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
	JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
	JitParam jit_do_soffset_index( QDP_get_global_cache().addJitParamBool( true ) );   // do soffset index

	std::vector<QDPCache::ArgKey> ids;
	ids.push_back( jit_ordered.get_id() );
	ids.push_back( jit_th_count.get_id() );
	ids.push_back( jit_start.get_id() );
	ids.push_back( jit_end.get_id() );
	ids.push_back( jit_do_soffset_index.get_id() );
	ids.push_back( MasterMap::Instance().getIdInner(s,offnode_maps) );
	ids.push_back( s.getIdMemberTable() );
	for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	  ids.push_back( addr_leaf.ids[i] );
 
	jit_launch(function,th_count,ids);
      }
      
      // 2nd call: face
      {
	ShiftPhase2 phase2;
	forEach(rhs, phase2 , NullCombine());

	int th_count = MasterMap::Instance().getCountFace(s,offnode_maps);
      
	JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
	JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
	JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
	JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
	JitParam jit_do_soffset_index( QDP_get_global_cache().addJitParamBool( true ) );   // do soffset index

	std::vector<QDPCache::ArgKey> ids;
	ids.push_back( jit_ordered.get_id() );
	ids.push_back( jit_th_count.get_id() );
	ids.push_back( jit_start.get_id() );
	ids.push_back( jit_end.get_id() );
	ids.push_back( jit_do_soffset_index.get_id() );
	ids.push_back( MasterMap::Instance().getIdFace(s,offnode_maps) );
	ids.push_back( s.getIdMemberTable() );
	for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
	  ids.push_back( addr_leaf.ids[i] );
 
	jit_launch(function,th_count,ids);
      }

      
    }


}





template<class T, class C1, class Op, class RHS>
void 
function_subtype_type_exec(JitFunction function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,C1 >& rhs, const Subset& s)
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
operator_type_subtype_exec(JitFunction function, OLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs, const Subset& s)
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
operator_subtype_subtype_exec(JitFunction function, OSubLattice<T>& dest, const Op& op, const QDPSubType<T1,OLattice<T1> >& rhs, const Subset& s)
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



  


template<class T, class T1, class Op, class RHS>
void 
function_lat_sca_subtype_exec(JitFunction function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
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





template<class T>
void 
function_zero_rep_exec(JitFunction function, OLattice<T>& dest, const Subset& s )
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  if (s.numSiteTable() < 1)
    return;

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());

  int th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();

  JitParam jit_ordered( QDP_get_global_cache().addJitParamBool( s.hasOrderedRep() ) );
  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  JitParam jit_start( QDP_get_global_cache().addJitParamInt( s.start() ) );
  JitParam jit_end( QDP_get_global_cache().addJitParamInt( s.end() ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_ordered.get_id() );
  ids.push_back( jit_th_count.get_id() );
  ids.push_back( jit_start.get_id() );
  ids.push_back( jit_end.get_id() );
  ids.push_back( s.getIdMemberTable() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



template<class T>
void function_zero_rep_subtype_exec(JitFunction function, OSubLattice<T>& dest, const Subset& s )
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
    return;
  }

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());

  JitParam jit_th_count( QDP_get_global_cache().addJitParamInt( th_count ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_th_count.get_id() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,th_count,ids);
}



template<class T>
void 
function_random_exec(JitFunction function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp)
{
  if (!s.hasOrderedRep())
    QDP_error_exit("random on subset with unordered representation not implemented");

  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());

  forEach(RNG::ran_seed, addr_leaf, NullCombine());
  forEach(seed_tmp, addr_leaf, NullCombine());
  forEach(RNG::ran_mult_n, addr_leaf, NullCombine());
  forEach(*RNG::lattice_ran_mult, addr_leaf, NullCombine());

  JitParam jit_lo( QDP_get_global_cache().addJitParamInt( s.start() ) );
  JitParam jit_hi( QDP_get_global_cache().addJitParamInt( s.end() ) );
  
  std::vector<QDPCache::ArgKey> ids;
  ids.push_back( jit_lo.get_id() );
  ids.push_back( jit_hi.get_id() );
  for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
    ids.push_back( addr_leaf.ids[i] );
 
  jit_launch(function,s.numSiteTable(),ids);
}


}

#endif
