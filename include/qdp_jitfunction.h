#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#define JIT_DO_MEMBER
//#undef JIT_DO_MEMBER


#include "qmp.h"

namespace QDP {





  template<class T, class T1, class T2, class C1, class C2>
  void 
  function_localInnerProduct_type_subtype_exec(CUfunction function, OSubLattice<T>& ret,
					       const QDPType<T1,C1> & l,const QDPSubType<T2,C2> & r,
					       const Subset& s)
  {
    int th_count = s.numSiteTable();

    if (th_count < 1) {
      //QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
      return;
    }

    void * site_table = QDP_get_global_cache().getDevicePtr( s.getIdSiteTable() );
    AddressLeaf addr_leaf(s);
    FnLocalInnerProduct op;
    forEach(ret, addr_leaf, NullCombine());
    AddOpAddress<FnLocalInnerProduct,AddressLeaf>::apply(op,addr_leaf);
    forEach(l, addr_leaf, NullCombine());
    forEach(r, addr_leaf, NullCombine());

    std::vector<void*> addr;
    addr.push_back( &th_count );
    addr.push_back( &site_table );

    for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
      addr.push_back( &addr_leaf.addr[i] );
    }
    jit_launch(function,th_count,addr);
  }

  
  
  template<class T, class T1, class T2, class C1, class C2>
  CUfunction
  function_localInnerProduct_type_subtype_build(OSubLattice<T>& ret,
						const QDPType<T1,C1> & l,const QDPSubType<T2,C2> & r)
  {
    typedef typename QDPType<T1,C1>::Subtype_t    LT;
    typedef typename QDPSubType<T2,C2>::Subtype_t RT;
    
    if (ptx_db::db_enabled) {
      CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (func)
	return func;
    }

    llvm_start_new_function();

    ParamRef p_th_count     = llvm_add_param<int>();
    ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

    ParamLeaf param_leaf;

    typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   ret_jit(forEach(ret, param_leaf, TreeCombine()));

    FnLocalInnerProduct op;
    auto op_jit = AddOpParam<FnLocalInnerProduct,ParamLeaf>::apply(op,param_leaf);

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





template<class T, class T1, class Op, class RHS>
CUfunction
function_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
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




template<class T, class T1, class Op, class RHS>
CUfunction
function_subtype_build(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
      return func;
  }

  llvm_start_new_function();

  ParamRef p_th_count     = llvm_add_param<int>();
  ParamRef p_site_table   = llvm_add_param<int*>();      // subset sitetable

  ParamLeaf param_leaf;

  typename LeafFunctor<OSubLattice<T>, ParamLeaf>::Type_t   dest_jit(forEach(dest, param_leaf, TreeCombine()));
  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
  typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t rhs_jit(forEach(rhs, param_leaf, TreeCombine()));
  
  llvm::Value * r_th_count     = llvm_derefParam( p_th_count );
  llvm::Value* r_idx_thread = llvm_thread_idx();

  llvm_cond_exit( llvm_ge( r_idx_thread , r_th_count ) );

  llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_table , r_idx_thread );

  op_jit( dest_jit.elem( JitDeviceLayout::Scalar , r_idx_thread ), // Coalesced
	  forEach(rhs_jit, ViewLeaf( JitDeviceLayout::Coalesced , r_idx_perm ), OpCombine()));

  return jit_function_epilogue_get_cuf("jit_eval_subtype_expr.ptx" , __PRETTY_FUNCTION__ );
}


  


template<class T, class T1, class Op, class RHS>
CUfunction
function_lat_sca_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
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



template<class T, class T1>
CUfunction
function_pokeSite_build( const OLattice<T>& dest , const OScalar<T1>& r  )
{
  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
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
function_pokeSite_exec(CUfunction function, OLattice<T>& dest, const OScalar<T1>& rhs, const multi1d<int>& coord )
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf(all);

  forEach(dest, addr_leaf, NullCombine());
  forEach(rhs, addr_leaf, NullCombine());

  int siteindex = Layout::linearSiteIndex(coord);
  int th_count = 1;

  std::vector<void*> addr;

  addr.push_back( &siteindex );
  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
  }

  jit_launch(function,th_count,addr);
}







template<class T>
CUfunction
function_zero_rep_build(OLattice<T>& dest)
{
  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
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
CUfunction
function_zero_rep_subtype_build(OSubLattice<T>& dest)
{
  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
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
CUfunction
function_random_build(OLattice<T>& dest , Seed& seed_tmp)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
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
CUfunction
function_gather_build( void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  if (ptx_db::db_enabled) {
    CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
    if (func)
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
  function_gather_exec( CUfunction function, void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs , const Subset& subset )
{

  AddressLeaf addr_leaf(subset);

  forEach(rhs, addr_leaf, NullCombine());

  //QDP_get_global_cache().printLockSets();

  // lo <= idx < hi
  int lo = 0;
  int hi = map.soffset(subset).size();

  //QDP_info("gather sites into send_buf lo=%d hi=%d",lo,hi);

  int soffsetsId = map.getSoffsetsId(subset);
  void * soffsetsDev = QDP_get_global_cache().getDevicePtr( soffsetsId );

  //QDP_get_global_cache().printLockSets();

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo =" << addr[0] << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi =" << addr[1] << "\n";

  addr.push_back( &soffsetsDev );
  //std::cout << "addr soffsetsDev =" << addr[3] << " " << soffsetsDev << "\n";

  addr.push_back( &send_buf );
  //std::cout << "addr send_buf =" << addr[4] << " " << send_buf << "\n";

  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr rhs =" << addr[addr.size()-1] << " " << addr_leaf.addr[i] << "\n";
  }

  jit_launch(function,hi-lo,addr);
}


namespace COUNT {
  extern int count;
}


template<class T, class T1, class Op, class RHS>
void 
function_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  //QDPIO::cout << __PRETTY_FUNCTION__ << "\n";

  // This should be supported
  //assert( s.hasOrderedRep() && "only ordered subsets are supported");

  ShiftPhase1 phase1(s);
  int offnode_maps = forEach(rhs, phase1 , BitOrCombine());
  //QDP_info("offnode_maps = %d",offnode_maps);


  void * idx_inner_dev = NULL;
  void * idx_face_dev = NULL;

  // lo <= idx < hi
  int start = s.start();
  int end = s.end();
  int th_count;
  bool ordered = s.hasOrderedRep();
  bool do_soffset_index;

  int innerCount, faceCount;

  if (offnode_maps > 0) {
    int innerId, faceId;
    innerId = MasterMap::Instance().getIdInner(s,offnode_maps);
    innerCount = MasterMap::Instance().getCountInner(s,offnode_maps);
    faceId = MasterMap::Instance().getIdFace(s,offnode_maps);
    faceCount = MasterMap::Instance().getCountFace(s,offnode_maps);
    idx_inner_dev = QDP_get_global_cache().getDevicePtr( innerId );
    idx_face_dev = QDP_get_global_cache().getDevicePtr( faceId );
    th_count = innerCount;
    do_soffset_index = true;
    //QDP_info("innerId = %d innerCount = %d faceId = %d  faceCount = %d",innerId,innerCount,faceId,faceCount);
    //QDPIO::cout << "innerId = "<< innerId <<" innerCount = "<<innerCount<<" faceId = "<<faceId<<"  faceCount = " <<faceCount << "\n";
  } else {
    th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
    do_soffset_index = false;
  }



  void * subset_member = QDP_get_global_cache().getDevicePtr( s.getIdMemberTable() );


  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());



  std::vector<void*> addr;

  /* if (COUNT::count == 12357) */
  /*   { */
  /*     QDPIO::cout << "ordered = " << ordered << "\n"; */
  /*     QDPIO::cout << "th_count = " << th_count << "\n"; */
  /*     QDPIO::cout << "start = " << start << "\n"; */
  /*     QDPIO::cout << "end = " << end << "\n"; */
  /*     QDPIO::cout << "do_soffset_index  = " << do_soffset_index  << "\n"; */
  /*     QDPIO::cout << "idx_inner_dev = " << idx_inner_dev << "\n"; */
  /*     QDPIO::cout << "subset_member = " << subset_member << "\n"; */
  /*   } */

  addr.push_back( &ordered );
  addr.push_back( &th_count );
  addr.push_back( &start );
  addr.push_back( &end );
  addr.push_back( &do_soffset_index );
  addr.push_back( &idx_inner_dev );
  addr.push_back( &subset_member );


  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

 
  jit_launch(function,th_count,addr);


  if (offnode_maps > 0) {
    ShiftPhase2 phase2;
    forEach(rhs, phase2 , NullCombine());

    th_count = faceCount;
    idx_inner_dev = idx_face_dev;

    jit_launch(function,th_count,addr);
  }
  /* ++COUNT::count; */
}



  
template<class T, class T1, class Op, class RHS>
void 
function_subtype_exec(CUfunction function, OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
    return;
  }

  void * site_table = QDP_get_global_cache().getDevicePtr( s.getIdSiteTable() );
  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  std::vector<void*> addr;
  addr.push_back( &th_count );
  addr.push_back( &site_table );

  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
  }
  jit_launch(function,th_count,addr);
}




  
template<class T, class T1, class Op, class RHS>
void 
function_lat_sca_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  forEach(rhs, addr_leaf, NullCombine());

  int start = s.start();
  int end = s.end();
  bool ordered = s.hasOrderedRep();
  int th_count = ordered ? s.numSiteTable() : Layout::sitesOnNode();

  void * subset_member = QDP_get_global_cache().getDevicePtr( s.getIdMemberTable() );

  std::vector<void*> addr;

  addr.push_back( &ordered );

  addr.push_back( &th_count );

  addr.push_back( &start );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &end );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  addr.push_back( &subset_member );
  //std::cout << "addr idx_inner_dev = " << addr[3] << " " << idx_inner_dev << "\n";

  //int addr_dest=addr.size();
  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  jit_launch(function,th_count,addr);
}






template<class T>
void 
function_zero_rep_exec(CUfunction function, OLattice<T>& dest, const Subset& s )
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf(s);

  forEach(dest, addr_leaf, NullCombine());

  int start = s.start();
  int end = s.end();
  bool ordered = s.hasOrderedRep();
  int th_count = ordered ? s.numSiteTable() : Layout::sitesOnNode();

  void * subset_member = QDP_get_global_cache().getDevicePtr( s.getIdMemberTable() );

  std::vector<void*> addr;

  addr.push_back( &ordered );
  //std::cout << "ordered = " << ordered << "\n";

  addr.push_back( &th_count );
  //std::cout << "thread_count = " << th_count << "\n";

  addr.push_back( &start );
  //std::cout << "start        = " << start << "\n";

  addr.push_back( &end );
  //std::cout << "end          = " << end << "\n";

  addr.push_back( &subset_member );
  //std::cout << "addr idx_inner_dev = " << addr[3] << " " << idx_inner_dev << "\n";

  //int addr_dest=addr.size();
  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }


  jit_launch(function,th_count,addr);
}



template<class T>
void function_zero_rep_subtype_exec(CUfunction function, OSubLattice<T>& dest, const Subset& s )
{
  int th_count = s.numSiteTable();

  if (th_count < 1) {
    QDPIO::cout << "skipping localInnerProduct since zero size subset on this MPI\n";
    return;
  }

  AddressLeaf addr_leaf(s);
  forEach(dest, addr_leaf, NullCombine());

  std::vector<void*> addr;
  addr.push_back( &th_count );

  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
  }
  jit_launch(function,th_count,addr);
}


  


template<class T>
void 
function_random_exec(CUfunction function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp)
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

  // lo <= idx < hi
  int lo = s.start();
  int hi = s.end();

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  //int addr_dest=addr.size();
  for(unsigned i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  jit_launch(function,s.numSiteTable(),addr);
}


}

#endif
