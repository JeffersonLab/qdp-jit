#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#define JIT_DO_MEMBER
//#undef JIT_DO_MEMBER


#include "qmp.h"

namespace QDP {


#if 0
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  jit_start_new_function();

  //function.setPrettyFunction(__PRETTY_FUNCTION__);

  llvm::Value * r_ordered      = llvm_add_param(  jit_llvm_builtin::i1 );
  llvm::Value * r_th_count     = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_start        = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_end          = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_do_site_perm = llvm_add_param(  jit_llvm_builtin::i1 );
  llvm::Value * r_no_site_perm = jit_ins_not( r_do_site_perm );

  llvm::Value * r_idx_thread = llvm_thread_idx();

  jit_ins_cond_exit( jit_ins_ge( r_idx_thread , r_th_count ) );

  jit_block_t block_no_site_perm_exit;
  jit_block_t block_no_site_perm;
  jit_block_t block_site_perm;
  jit_block_t block_add_start;

  llvm::Value * r_idx_perm_phi0;
  llvm::Value * r_idx_perm_phi1;
  jit_ins_branch( r_no_site_perm , block_no_site_perm , block_site_perm );
  {
    jit_ins_start_block(block_site_perm);
    r_idx_perm_phi0 = jit_int_array_indirection( r_idx_thread , jit_llvm_builtin::i32 ); // PHI 0
    jit_ins_branch( block_no_site_perm_exit );
  }
  jit_ins_start_block(block_no_site_perm);
  {
    llvm::Value * r_not_ordered = jit_ins_not(r_ordered);
    jit_ins_branch( r_not_ordered , block_no_site_perm_exit , block_add_start );
    jit_ins_start_block(block_add_start);
    r_idx_perm_phi1 = jit_ins_add( r_idx_thread , r_start ); // PHI 1
    jit_ins_branch( block_no_site_perm_exit );
  }
  jit_ins_start_block(block_no_site_perm_exit);

  llvm::Value * r_idx = jit_ins_phi( r_idx_perm_phi0 , block_site_perm , r_idx_perm_phi1 , block_add_start );


  jit_block_t block_ordered;
  jit_block_t block_not_ordered;
  jit_block_t block_ordered_exit;
  jit_ins_branch( r_ordered , block_ordered , block_not_ordered );
  {
    jit_ins_start_block(block_not_ordered);
    llvm::Value * r_ismember     = jit_int_array_indirection( r_idx , jit_llvm_builtin::i1 );
    llvm::Value * r_ismember_not = jit_ins_not( r_ismember );
    jit_ins_cond_exit( r_ismember_not );
    jit_ins_branch( block_ordered_exit );
  }
  {
    jit_ins_start_block(block_ordered);
    jit_ins_cond_exit( jit_ins_gt( r_idx , r_end ) );
    jit_ins_cond_exit( jit_ins_lt( r_idx , r_start ) );
    jit_ins_branch( block_ordered_exit );
  }
  jit_ins_start_block(block_ordered_exit);
#endif



template<class T, class T1, class Op, class RHS>
CUfunction
function_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  llvm::Value* r_ordered      = llvm_add_param<bool>();
  llvm::Value* r_th_count     = llvm_add_param<int>();
  llvm::Value* r_start        = llvm_add_param<int>();
  llvm::Value* r_end          = llvm_add_param<int>();
  llvm::Value* r_do_site_perm = llvm_add_param<bool>();

  llvm::Value* r_no_site_perm = llvm_not( r_do_site_perm );  

  llvm::Value* r_idx_thread = llvm_thread_idx();

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
    r_idx_perm_phi0 = llvm_array_type_indirection<int*>( r_idx_thread ); // PHI 0
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
    llvm::Value* r_ismember     = llvm_array_type_indirection<bool*>( r_idx );
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

  ParamLeaf param_leaf(  r_idx );
  //ParamLeaf param_leaf_indexed(  param_leaf.getParamIndexFieldAndOption() );  // Optional soffset (inner/face)
  //function.addParamMemberArray( param_leaf.r_idx ); // Subset member
  
  // Destination
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  //FuncRet_t dest_jit(forEach(dest, param_leaf_indexed, TreeCombine()));
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);
  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  //View_t rhs_view(forEach(rhs, param_leaf_indexed, TreeCombine()));
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));
  //printme<View_t>();

  op_jit(dest_jit.elem( JitDeviceLayout::Coalesced ), forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced ), OpCombine()));

  llvm_exit();

  return llvm_get_cufunction("eval.ptx");
}


template<class T, class T1, class Op, class RHS>
CUfunction
function_lat_sca_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  CUfunction func;

  jit_start_new_function();

  //function.setPrettyFunction(__PRETTY_FUNCTION__);

  llvm::Value * r_ordered      = llvm_add_param(  jit_llvm_builtin::i1 );
  llvm::Value * r_th_count     = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_start        = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_end          = llvm_add_param(  jit_llvm_builtin::i32 );

  llvm::Value * r_idx_thread = llvm_thread_idx();

  jit_ins_exit( jit_ins_ge( r_idx_thread , r_th_count ) );

  llvm::Value * r_idx = r_idx_thread;

  jit_block_t block_ordered;
  jit_block_t block_ordered_exit;
  jit_ins_branch( block_ordered , r_ordered );
  {
    llvm::Value * r_member = llvm_add_param(  jit_llvm_builtin::u64 );  // Subset
    llvm::Value * r_member_addr        = jit_ins_add( r_member , r_idx );   // I don't have to multiply with wordsize, since 1
    llvm::Value * r_ismember           = jit_ins_load ( r_member_addr , 0 , jit_llvm_builtin::i1 );
    llvm::Value * r_ismember_not       = jit_ins_not( r_ismember );
    jit_ins_exit( r_ismember_not );
    jit_ins_branch( block_ordered_exit );
  }
  jit_ins_start_block(block_ordered);
  {
    r_idx = jit_ins_add( r_idx_thread , r_start );
  }
  jit_ins_start_block(block_ordered_exit);

  ParamLeaf param_leaf(  r_idx );
  
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  op_jit(dest_jit.elem( JitDeviceLayout::Coalesced ), forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar ), OpCombine()));

  return jit_get_cufunction("ll_lat_sca.ll");
#endif
}






template<class T>
CUfunction
function_zero_rep_build(OLattice<T>& dest)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0

  CUfunction func;

  jit_start_new_function();

  llvm::Value * r_ordered      = llvm_add_param(  jit_llvm_builtin::i1 );
  llvm::Value * r_th_count     = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_start        = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_end          = llvm_add_param(  jit_llvm_builtin::i32 );

  llvm::Value * r_idx_thread = llvm_thread_idx();

  jit_ins_exit( jit_ins_ge( r_idx_thread , r_th_count ) );

  llvm::Value * r_idx = r_idx_thread;

  jit_block_t block_ordered;
  jit_block_t block_ordered_exit;
  jit_ins_branch( block_ordered , r_ordered );
  {
    llvm::Value * r_member = llvm_add_param(  jit_llvm_builtin::u64 );  // Subset
    llvm::Value * r_member_addr        = jit_ins_add( r_member , r_idx );   // I don't have to multiply with wordsize, since 1
    llvm::Value * r_ismember           = jit_ins_load ( r_member_addr , 0 , jit_llvm_builtin::i1 );
    llvm::Value * r_ismember_not       = jit_ins_not( r_ismember );
    jit_ins_exit( r_ismember_not );
    jit_ins_branch( block_ordered_exit );
  }
  jit_ins_start_block(block_ordered);
  {
    r_idx = jit_ins_add( r_idx_thread , r_start );
  }
  jit_ins_start_block(block_ordered_exit);

  ParamLeaf param_leaf(  r_idx );

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced) );

  return jit_get_cufunction("ll_zero.ll");
#endif
}







template<class T, class T1, class Op, class RHS>
CUfunction
function_sca_sca_build(OScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0

  CUfunction func;

  jit_start_new_function();

  llvm::Value * r_idx = llvm_thread_idx();

  ParamLeaf param_leaf(  r_idx );

  typedef typename LeafFunctor<OScalar<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  //View_t rhs_view(forEach(rhs, param_leaf_indexed, TreeCombine()));
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  //printme<View_t>();

  op_jit(dest_jit.elem( JitDeviceLayout::Scalar ), forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar ), OpCombine()));

  return jit_get_cufunction("ll_sca_sca.ll");
#endif
}




template<class T>
CUfunction
function_random_build(OLattice<T>& dest , Seed& seed_tmp)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  jit_start_new_function();

  // No member possible here.
  // If thread exists due to non-member
  // it possibly can't store the new seed at the end.

  llvm::Value * r_lo     = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_hi     = llvm_add_param(  jit_llvm_builtin::i32 );

  llvm::Value * r_idx_thread = llvm_thread_idx();

  llvm::Value * r_out_of_range       = jit_ins_gt( r_idx_thread , jit_ins_sub( r_hi , r_lo ) );
  jit_ins_exit(  r_out_of_range );

  llvm::Value * r_idx = jit_ins_add( r_lo , r_idx_thread );

  ParamLeaf param_leaf(  r_idx );

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

  PSeedREG seed_reg;
  PSeedREG skewed_seed_reg;
  PSeedREG ran_mult_n_reg;
  PSeedREG lattice_ran_mult_reg;

  seed_reg.setup( ran_seed_jit.elem() );

  lattice_ran_mult_reg.setup( lattice_ran_mult_jit.elem( JitDeviceLayout::Coalesced ) );

  skewed_seed_reg = seed_reg * lattice_ran_mult_reg;

  ran_mult_n_reg.setup( ran_mult_n_jit.elem() );

  fill_random( dest_jit.elem(JitDeviceLayout::Coalesced) , seed_reg , skewed_seed_reg , ran_mult_n_reg );

  llvm::Value * r_no_save = jit_ins_ne( r_idx_thread , llvm::Value *(0) );

  jit_block_t block_nosave;
  jit_ins_branch(  block_nosave , r_no_save );
  seed_tmp_jit.elem() = seed_reg;
  jit_ins_start_block(  block_nosave );

  return jit_get_cufunction("ll_random.ll");
#endif
}








template<class T, class T1, class RHS>
CUfunction
function_gather_build( void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  jit_start_new_function();

  llvm::Value * r_lo     = llvm_add_param(  jit_llvm_builtin::i32 );
  llvm::Value * r_hi     = llvm_add_param(  jit_llvm_builtin::i32 );

  llvm::Value * r_idx = llvm_thread_idx();  

  llvm::Value * r_out_of_range       = jit_ins_ge( r_idx , r_hi );
  jit_ins_exit(  r_out_of_range );

  llvm::Value * r_perm_array_addr      = llvm_add_param(  jit_llvm_builtin::u64 );  // Site permutation array
  llvm::Value * r_idx_mul_4            = jit_ins_mul( r_idx , llvm::Value *(4) );
  llvm::Value * r_perm_array_addr_load = jit_ins_add( r_perm_array_addr , r_idx_mul_4 );
  llvm::Value * r_idx_perm             = jit_ins_load ( r_perm_array_addr_load , 0 , jit_llvm_builtin::i32 );

  llvm::Value * r_gather_buffer        = llvm_add_param(  jit_llvm_builtin::u64 );  // Gather buffer

  //ParamLeaf param_leaf_idx(  r_idx );
  ParamLeaf param_leaf_idx_perm(  r_idx_perm );
  
  // ParamLeaf param_leaf_0(  function.getRegIdx() , Jit::LatticeLayout::SCAL );
  // ParamLeaf param_leaf_soffset(  param_leaf_0.getParamIndexFieldAndOption() , Jit::LatticeLayout::COAL );

  // Destination
  typedef typename JITType< OLattice<T> >::Type_t DestView_t;

  DestView_t dest_jit(  r_gather_buffer , r_idx );

		       // param_leaf_0.getParamLattice( JITType<T>::Type_t::Size_t * WordSize<T>::Size ) ,
		       // Jit::LatticeLayout::SCAL );

  // typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  // FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view( forEach( rhs , param_leaf_idx_perm , TreeCombine() ) );

  //printme<View_t>();

  OpAssign()( dest_jit.elem( JitDeviceLayout::Scalar ) , 
	      forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced ) , OpCombine() ) );

  return jit_get_cufunction("ll_gather.ll");
#endif
}


template<class T1, class RHS>
void
function_gather_exec( CUfunction function, void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  AddressLeaf addr_leaf;

  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  //QDPCache::Instance().printLockSets();

  // lo <= idx < hi
  int lo = 0;
  int hi = map.soffset().size();

  //QDP_info("gather sites into send_buf lo=%d hi=%d",lo,hi);

  int soffsetsId = map.getSoffsetsId();
  void * soffsetsDev = QDPCache::Instance().getDevicePtr( soffsetsId );

#if 0
  int size = QDPCache::Instance().getSize( soffsetsId );
  std::cout << "allocating host memory for soffset, size = " << size << "\n";
  unsigned char * soff_host = new unsigned char[ size ];
  std::cout << "copying...\n";
  CudaMemcpyD2H(soff_host,soffsetsDev,size);
  for(int i=0;i<size/4;i++)
    std::cout << ((int*)soff_host)[i] << " ";
  std::cout << "\n";
  delete[] soff_host;
#endif

  //QDPCache::Instance().printLockSets();

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo =" << addr[0] << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi =" << addr[1] << "\n";

  addr.push_back( &soffsetsDev );
  //std::cout << "addr soffsetsDev =" << addr[3] << " " << soffsetsDev << "\n";

  addr.push_back( &send_buf );
  //std::cout << "addr send_buf =" << addr[4] << " " << send_buf << "\n";

  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr rhs =" << addr[addr.size()-1] << " " << addr_leaf.addr[i] << "\n";
  }


  static std::map<int,int> threadsPerBlock;

  if (!threadsPerBlock[hi-lo]) {
    threadsPerBlock[hi-lo] = jit_autotuning(function,lo,hi,&addr[0]);
  } else {
    //QDP_info_primary("Previous gather_function auto-tuning result = %d",threadsPerBlock);
  }

  kernel_geom_t now = getGeom( hi-lo , threadsPerBlock[hi-lo] );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock[hi-lo],1,1,    0, 0, &addr[0] , 0);
#endif
}



template<class T, class T1, class Op, class RHS>
void 
function_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");

  //  std::cout << "function_exec 0\n";

  ShiftPhase1 phase1;
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
    innerId = MasterMap::Instance().getIdInner(offnode_maps);
    innerCount = MasterMap::Instance().getCountInner(offnode_maps);
    faceId = MasterMap::Instance().getIdFace(offnode_maps);
    faceCount = MasterMap::Instance().getCountFace(offnode_maps);
    idx_inner_dev = QDPCache::Instance().getDevicePtr( innerId );
    idx_face_dev = QDPCache::Instance().getDevicePtr( faceId );
    th_count = innerCount;
    do_soffset_index = true;
    //QDP_info("innerId = %d innerCount = %d faceId = %d  faceCount = %d",innerId,innerCount,faceId,faceCount);
  } else {
    th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();
    do_soffset_index = false;
  }



  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );


  // bool * member = new bool[QDPCache::Instance().getSize( s.getIdMemberTable() ) / sizeof(bool) ];
  // CudaMemcpyD2H( member , subset_member , QDPCache::Instance().getSize( s.getIdMemberTable() ) );
  // int co=0;
  // for (int i=0;i<hi;i++)
  //   if (member[i])
  //     co++;
  // std::cout << "member true = " << co << "   of " << hi << "\n";
  

  

  AddressLeaf addr_leaf;

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());



  std::vector<void*> addr;


  addr.push_back( &ordered );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  addr.push_back( &th_count );
  //std::cout << "thread_count = " << th_count << "\n";

  addr.push_back( &start );
  //std::cout << "start        = " << start << "\n";

  addr.push_back( &end );
  //std::cout << "end          = " << end << "\n";

  addr.push_back( &do_soffset_index );
  //std::cout << "addr do_soffset_index =" << addr[2] << " " << do_soffset_index << "\n";

  addr.push_back( &idx_inner_dev );
  //std::cout << "addr idx_inner_dev = " << addr[3] << " " << idx_inner_dev << "\n";

  addr.push_back( &subset_member );
  //std::cout << "addr subset_dev (member_array) = " << addr[3] << " " << subset_member << "\n";

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  static std::map<int,int> threadsPerBlock;

  if (!threadsPerBlock[th_count]) {
    // Auto tuning
    // Fist get a data field of the same size as "dest" 
    // where it's safe to do autotuning on.
    int tmpId = QDPCache::Instance().registrate( QDPCache::Instance().getSize( dest.getId() ) , 1 , NULL );
    void * devPtr = QDPCache::Instance().getDevicePtr( tmpId );
    //QDPCache::Instance().printLockSets();
    addr[addr_dest] = &devPtr;

    threadsPerBlock[th_count] = jit_autotuning(function,0,th_count,&addr[0]);

    // Restore original "dest" device address
    addr[addr_dest] = &addr_leaf.addr[0];
    QDPCache::Instance().signoff( tmpId );
    //QDPCache::Instance().printLockSets();

  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( th_count , threadsPerBlock[th_count] );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock[th_count],1,1,    0, 0, &addr[0] , 0);

  if (offnode_maps > 0) {
    ShiftPhase2 phase2;
    forEach(rhs, phase2 , NullCombine());

    th_count = faceCount;
    idx_inner_dev = idx_face_dev;

    if (!threadsPerBlock[th_count]) {
      int tmpId = QDPCache::Instance().registrate( QDPCache::Instance().getSize( dest.getId() ) , 1 , NULL );
      void * devPtr = QDPCache::Instance().getDevicePtr( tmpId );
      addr[addr_dest] = &devPtr;
      threadsPerBlock[th_count] = jit_autotuning(function,0,th_count,&addr[0]);
      addr[addr_dest] = &addr_leaf.addr[0];
      QDPCache::Instance().signoff( tmpId );
    }
    now = getGeom( th_count , threadsPerBlock[th_count] );                                  
    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock[th_count],1,1,    0, 0, &addr[0] , 0);
  }
}

template<class T, class T1, class Op, class RHS>
void 
function_lat_sca_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  int start = s.start();
  int end = s.end();
  bool ordered = s.hasOrderedRep();
  int th_count = ordered ? s.numSiteTable() : Layout::sitesOnNode();

  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

  std::vector<void*> addr;

  addr.push_back( &ordered );

  addr.push_back( &th_count );

  addr.push_back( &start );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &end );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  addr.push_back( &subset_member );
  //std::cout << "addr idx_inner_dev = " << addr[3] << " " << idx_inner_dev << "\n";

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    // Auto tuning

    // Fist get a data field of the same size as "dest" where we can play on
    // (in case the final operator is an OpAddAssign, etc.)
    int tmpId = QDPCache::Instance().registrate( QDPCache::Instance().getSize( dest.getId() ) , 1 , NULL );
    void * devPtr = QDPCache::Instance().getDevicePtr( tmpId );
    //QDPCache::Instance().printLockSets();
    addr[addr_dest] = &devPtr;

    threadsPerBlock = jit_autotuning(function,0,th_count,&addr[0]);

    // Restore original "dest" device address
    addr[addr_dest] = &addr_leaf.addr[0];
    QDPCache::Instance().signoff( tmpId );
    //QDPCache::Instance().printLockSets();

  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( th_count , threadsPerBlock );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);
#endif
}






template<class T>
void 
function_zero_rep_exec(CUfunction function, OLattice<T>& dest, const Subset& s )
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

  int start = s.start();
  int end = s.end();
  bool ordered = s.hasOrderedRep();
  int th_count = ordered ? s.numSiteTable() : Layout::sitesOnNode();

  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

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

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    threadsPerBlock = jit_autotuning(function,0,th_count,&addr[0]);
  }

  kernel_geom_t now = getGeom( th_count , threadsPerBlock );
  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);
#endif
}










template<class T, class T1, class Op, class RHS>
void 
function_sca_sca_exec(CUfunction function, OScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  AddressLeaf addr_leaf;

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  std::vector<void*> addr;

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  CudaLaunchKernel(function,   1,1,1,    1,1,1,    0, 0, &addr[0] , 0);
#endif
}




template<class T>
void 
function_random_exec(CUfunction function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
  if (!s.hasOrderedRep())
    QDP_error_exit("random on subset with unordered representation not implemented");

  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

  int junk_1 = forEach(RNG::ran_seed, addr_leaf, NullCombine());
  int junk_2 = forEach(seed_tmp, addr_leaf, NullCombine());
  int junk_3 = forEach(RNG::ran_mult_n, addr_leaf, NullCombine());
  int junk_4 = forEach(*RNG::lattice_ran_mult, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = s.start();
  int hi = s.end();

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    // Auto tuning
    threadsPerBlock = jit_autotuning(function,0,s.numSiteTable(),&addr[0]);
  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( s.numSiteTable() , threadsPerBlock );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);
#endif
}


}

#endif
