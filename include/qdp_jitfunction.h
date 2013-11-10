#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#define JIT_DO_MEMBER
//#undef JIT_DO_MEMBER


#include "qmp.h"

namespace QDP {



template<class T, class T1, class Op, class RHS>
void *
function_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  JitMainLoop loop;

  //ParamRef p_site_table   = llvm_add_param<int*>();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  IndexDomainVector idx = loop.getIdx();

  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , idx ), 
   	  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , idx ), OpCombine()));

  loop.done();

  return jit_function_epilogue_get("jit_eval.ptx");
}



template<class T, class T1, class Op, class RHS>
void 
function_exec(void * function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  /* ShiftPhase1 phase1; */
  /* int offnode_maps = forEach(rhs, phase1 , BitOrCombine()); */
  
#ifdef LLVM_DEBUG
  QDP_info("offnode_maps = %d",offnode_maps);
#endif

  /* ShiftPhase2 phase2; */
  /* forEach(rhs, phase2 , NullCombine()); */

  AddressLeaf addr_leaf;
  jit_get_empty_arguments(addr_leaf);

  addr_leaf.setOrdered( s.hasOrderedRep() );
  addr_leaf.setStart( s.start() );

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

#ifdef LLVM_DEBUG
  std::cout << "calling eval(Lattice,Lattice).. " << addr_leaf.addr.size() << "\n";  
#endif

  jit_dispatch(function,s.numSiteTable(),addr_leaf);
}





template<class T, class T1, class Op, class RHS>
void *
function_lat_sca_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  JitMainLoop loop;


  //ParamRef p_ordered      = llvm_add_param<bool>();
  //ParamRef p_th_count     = llvm_add_param<int>();
  //  ParamRef p_start        = llvm_add_param<int>();
  //ParamRef p_member_array = llvm_add_param<bool*>();

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  // llvm::Value * r_start        = llvm_derefParam( p_start );

  //llvm::Value * r_idx = loop.getIdx();
  IndexDomainVector idx = loop.getIdx();

  op_jit( dest_jit.elem( JitDeviceLayout::Coalesced , idx ), 
   	  forEach(rhs_view, ViewLeaf( JitDeviceLayout::Scalar , idx ), OpCombine()));

  loop.done();

  return jit_function_epilogue_get("jit_lat_sca.ptx");
}




template<class T, class T1, class Op, class RHS>
void 
function_lat_sca_exec(void* function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;
  jit_get_empty_arguments(addr_leaf);

  addr_leaf.setOrdered( s.hasOrderedRep() );
  addr_leaf.setStart( s.start() );

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  int th_count = s.hasOrderedRep() ? s.numSiteTable() : Layout::sitesOnNode();

#ifdef LLVM_DEBUG
  std::cout << "calling eval(Lattice,Scalar)..\n";
#endif

  jit_dispatch(function,th_count,addr_leaf);

  // void (*FP)(void*) = (void (*)(void*))(intptr_t)function;

  // std::cout << "calling eval(Lattice,Scalar)..\n";
  // FP( addr_leaf.addr.data() );
  // std::cout << "..done\n";

}







template<class T>
void *
function_zero_rep_build(OLattice<T>& dest)
{
  JitMainLoop loop;

  ParamLeaf param_leaf;

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  //llvm::Value * r_idx = loop.getIdx();
  IndexDomainVector idx = loop.getIdx();

  zero_rep( dest_jit.elem(JitDeviceLayout::Coalesced,idx) );

  loop.done();

  return jit_function_epilogue_get("jit_zero.ptx");
}





template<class T>
void 
function_zero_rep_exec(void * function, OLattice<T>& dest, const Subset& s )
{
  AddressLeaf addr_leaf;
  jit_get_empty_arguments(addr_leaf);

  addr_leaf.setOrdered( s.hasOrderedRep() );
  addr_leaf.setStart( s.start() );

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

#ifdef LLVM_DEBUG
  std::cout << "calling zero_rep(Lattice,Subset)..\n";
#endif

  jit_dispatch( function , s.numSiteTable() , addr_leaf );
}








#if 0
template<class T>
CUfunction
function_random_build(OLattice<T>& dest , Seed& seed_tmp)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

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

  return jit_function_epilogue_get_cuf("jit_random.ptx");
}
#endif




template<class T, class T1, class RHS>
void *
function_gather_build( void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
#ifdef LLVM_DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  typedef typename WordType<T1>::Type_t WT;

  JitMainLoop loop;

  // ParamRef p_lo      = llvm_add_param<int>();
  // ParamRef p_hi      = llvm_add_param<int>();

  ParamRef p_soffset = llvm_add_param<int*>();
  ParamRef p_sndbuf  = llvm_add_param<WT*>();

  ParamLeaf param_leaf;

  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view( forEach( rhs , param_leaf , TreeCombine() ) );

  typedef typename JITType< OLattice<T> >::Type_t DestView_t;
  DestView_t dest_jit( p_sndbuf );

  //llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

  //llvm::Value * r_idx = loop.getIdx();
  IndexDomainVector idx = loop.getIdx();

  llvm::Value * r_idx_site = llvm_array_type_indirection( p_soffset , get_index_from_index_vector(idx) );

  IndexDomainVector idx_vec_gather = get_index_vector_from_index( r_idx_site );

  OpAssign()( dest_jit.elem( JitDeviceLayout::Scalar , idx ) , 
	      forEach(rhs_view, ViewLeaf( JitDeviceLayout::Coalesced , idx_vec_gather ) , OpCombine() ) );

  loop.done();

  return jit_function_epilogue_get("jit_gather.ll");
}








template<class T1, class RHS>
void
function_gather_exec( void * function, void * send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  AddressLeaf addr_leaf;
  jit_get_empty_arguments(addr_leaf);

  AddressLeaf::Types t;
  t.ptr = const_cast<int*>(map.soffset().slice());
  addr_leaf.addr.push_back(t);

  t.ptr = send_buf;
  addr_leaf.addr.push_back(t);

  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  //QDP_info("gather sites into send_buf lo=%d hi=%d",lo,hi);

#ifdef LLVM_DEBUG
  std::cout << "calling gather.. " << addr_leaf.addr.size() << "\n";  
#endif
  jit_dispatch( function , map.soffset().size() , addr_leaf);
}







#if 0
template<class T>
void 
function_random_exec(CUfunction function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp)
{
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

  jit_launch(function,s.numSiteTable(),addr);
}
#endif


}

#endif
