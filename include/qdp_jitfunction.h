#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#define JIT_DO_MEMBER
//#undef JIT_DO_MEMBER


#include "qmp.h"

namespace QDP {


template<class T, class T1, class Op, class RHS>
CUfunction
function_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  const char * fname = "ptx_eval.ptx";
  jit_function_t function = jit_create_function( fname );

  //function.setPrettyFunction(__PRETTY_FUNCTION__);

  jit_value_t r_lo     = jit_add_param( function , jit_ptx_type::s32 );
  jit_value_t r_hi     = jit_add_param( function , jit_ptx_type::s32 );

  jit_value_t r_idx = jit_geom_get_linear_th_idx( function );  

  jit_value_t r_out_of_range       = jit_ins_ge( r_idx , r_hi );
  jit_ins_exit( function , r_out_of_range );


  jit_value_t r_do_site_perm         = jit_add_param( function , jit_ptx_type::s32 ); // Site permutation?, for inner sites
  jit_value_t r_do_site_perm_pred    = jit_ins_ne( r_do_site_perm , jit_val_create_const_int(0) );
  jit_value_t r_perm_array_addr      = jit_add_param( function , jit_ptx_type::u64 );  // Site permutation array
  jit_value_t r_idx_mul_4            = jit_ins_mul( r_idx , jit_val_create_const_int(4)  , r_do_site_perm_pred);
  jit_value_t r_perm_array_addr_load = jit_ins_add( r_perm_array_addr , r_idx_mul_4      , r_do_site_perm_pred);
  jit_value_t r_idx_perm             = jit_ins_load ( r_perm_array_addr_load , 0 , jit_ptx_type::s32 , r_do_site_perm_pred);
  jit_ins_mov_no_create( r_idx , r_idx_perm                                              , r_do_site_perm_pred);


  jit_value_t r_member = jit_add_param( function , jit_ptx_type::u64 );  // Subset
  jit_value_t r_member_addr        = jit_ins_add( r_member , r_idx );   // I don't have to multiply with wordsize, since 1
  jit_value_t r_ismember_u8        = jit_ins_load ( r_member_addr , 0 , jit_ptx_type::u8 );
  jit_value_t r_ismember_u32       = jit_val_create_convert( function , jit_ptx_type::u32 , r_ismember_u8 );
  jit_value_t r_ismember_pred_addr = jit_ins_eq( r_ismember_u32 , jit_val_create_const_int(0) );
  jit_ins_exit( function , r_ismember_pred_addr );


  ParamLeaf param_leaf( function , r_idx );
  //ParamLeaf param_leaf_indexed( function , param_leaf.getParamIndexFieldAndOption() );  // Optional soffset (inner/face)
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

  op_jit(dest_jit.elem( QDPTypeJITBase::Coalesced ), forEach(rhs_view, ViewLeaf( QDPTypeJITBase::Coalesced ), OpCombine()));

#if 1
  if (Layout::primaryNode())
    function->write();
#endif     
 
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad( &cuModule , fname );
  if (ret) QDP_error_exit( "Error loading CUDA module '%s'" , fname );

  ret = cuModuleGetFunction(&func, cuModule, "function");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}


template<class T, class T1, class Op, class RHS>
CUfunction
function_lat_sca_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  const char * fname = "ptx_lat_sca.ptx";
  jit_function_t function = jit_create_function( fname );

  //function.setPrettyFunction(__PRETTY_FUNCTION__);

  jit_value_t r_lo     = jit_add_param( function , jit_ptx_type::s32 );
  jit_value_t r_hi     = jit_add_param( function , jit_ptx_type::s32 );
#ifdef JIT_DO_MEMBER
  jit_value_t r_member = jit_add_param( function , jit_ptx_type::u64 );  // Subset
#endif

  jit_value_t r_idx = jit_geom_get_linear_th_idx( function );  

  jit_value_t r_out_of_range       = jit_ins_ge( r_idx , r_hi );
  jit_ins_exit( function , r_out_of_range );

#ifdef JIT_DO_MEMBER
  jit_value_t r_member_addr        = jit_ins_add( r_member , r_idx );   // I don't have to multiply with wordsize, since 1
  jit_value_t r_ismember_u8        = jit_ins_load ( r_member_addr , 0 , jit_ptx_type::u8 );
  jit_value_t r_ismember_u32       = jit_val_create_convert( function , jit_ptx_type::u32 , r_ismember_u8 );
  jit_value_t r_ismember_pred_addr = jit_ins_eq( r_ismember_u32 , jit_val_create_const_int(0) );
  jit_ins_exit( function , r_ismember_pred_addr );
#endif

  ParamLeaf param_leaf( function , r_idx );
  
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  auto op_jit = AddOpParam<Op,ParamLeaf>::apply(op,param_leaf);

  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  op_jit(dest_jit.elem( QDPTypeJITBase::Coalesced ), forEach(rhs_view, ViewLeaf( QDPTypeJITBase::Scalar ), OpCombine()));

#if 1
  if (Layout::primaryNode())
    function->write();
#endif     
 
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad( &cuModule , fname );
  if (ret) QDP_error_exit( "Error loading CUDA module '%s'" , fname );

  ret = cuModuleGetFunction(&func, cuModule, "function");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}






template<class T>
CUfunction
function_zero_rep_build(OLattice<T>& dest)
{
  CUfunction func;

  const char * fname = "ptx_zero.ptx";
  jit_function_t function = jit_create_function( fname );

  jit_value_t r_lo     = jit_add_param( function , jit_ptx_type::s32 );
  jit_value_t r_hi     = jit_add_param( function , jit_ptx_type::s32 );
#ifdef JIT_DO_MEMBER
  jit_value_t r_member = jit_add_param( function , jit_ptx_type::u64 );  // Subset
#endif

  jit_value_t r_idx = jit_geom_get_linear_th_idx( function );

  jit_value_t r_out_of_range       = jit_ins_ge( r_idx , r_hi );
  jit_ins_exit( function , r_out_of_range );

#ifdef JIT_DO_MEMBER
  jit_value_t r_member_addr        = jit_ins_add( r_member , r_idx );   // I don't have to multiply with wordsize, since 1
  jit_value_t r_ismember_u8        = jit_ins_load ( r_member_addr , 0 , jit_ptx_type::u8 );
  jit_value_t r_ismember_u32       = jit_val_create_convert( function , jit_ptx_type::u32 , r_ismember_u8 );
  jit_value_t r_ismember_pred_addr = jit_ins_eq( r_ismember_u32 , jit_val_create_const_int(0) );
  jit_ins_exit( function , r_ismember_pred_addr );
#endif

  ParamLeaf param_leaf( function , r_idx );

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  zero_rep( dest_jit.elem(QDPTypeJITBase::Coalesced) );

  if (Layout::primaryNode())
    function->write();
      
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad( &cuModule , fname );
  if (ret) QDP_error_exit("Error loading CUDA module '%s'" , fname );

  ret = cuModuleGetFunction(&func, cuModule, "function");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  return func;
}





#if 0

template<class T, class T1, class Op, class RHS>
CUfunction
function_sca_sca_build(OScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  std::string fname("ptx_sca_sca.ptx");
  Jit function(fname.c_str(),"func");

  function.setPrettyFunction(__PRETTY_FUNCTION__);

  //std::cout << "function = " << (void*)&function <<"\n";

  ParamLeaf param_leaf(function,function.getRegIdx() , Jit::LatticeLayout::SCAL );
  //ParamLeaf param_leaf_indexed( function , param_leaf.getParamIndexFieldAndOption() , Jit::LatticeLayout::COAL);
  //function.addParamMemberArray( param_leaf.r_idx );

  // Destination
  typedef typename LeafFunctor<OScalar<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OScalar<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  //printme<View_t>();

  op(dest_jit.elem( 0 ), forEach(rhs_view, ViewLeaf(0), OpCombine()));

#if 1
  if (Layout::primaryNode())
    function.write();
#endif     
 
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, fname.c_str() );
  if (ret) QDP_error_exit("Error loading CUDA module '%s'",fname.c_str());

  ret = cuModuleGetFunction(&func, cuModule, "func");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}
#endif



template<class T>
CUfunction
function_random_build(OLattice<T>& dest , Seed& seed_tmp)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  const char * fname = "ptx_random.ptx";
  jit_function_t function = jit_create_function( fname );

  jit_value_t r_lo     = jit_add_param( function , jit_ptx_type::s32 );
  jit_value_t r_hi     = jit_add_param( function , jit_ptx_type::s32 );
#ifdef JIT_DO_MEMBER
  jit_value_t r_member = jit_add_param( function , jit_ptx_type::u64 );  // Subset
#endif

  jit_value_t r_idx = jit_geom_get_linear_th_idx( function );  

  jit_value_t r_out_of_range       = jit_ins_ge( r_idx , r_hi );
  jit_ins_exit( function , r_out_of_range );

#ifdef JIT_DO_MEMBER
  jit_value_t r_member_addr        = jit_ins_add( r_member , r_idx );   // I don't have to multiply with wordsize, since 1
  jit_value_t r_ismember_u8        = jit_ins_load ( r_member_addr , 0 , jit_ptx_type::u8 );
  jit_value_t r_ismember_u32       = jit_val_create_convert( function , jit_ptx_type::u32 , r_ismember_u8 );
  jit_value_t r_ismember_pred_addr = jit_ins_eq( r_ismember_u32 , jit_val_create_const_int(0) );
  jit_ins_exit( function , r_ismember_pred_addr );
#endif

  ParamLeaf param_leaf( function , r_idx );

  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // RNG::ran_seed
  typedef typename LeafFunctor<Seed, ParamLeaf>::Type_t  SeedJIT;
  typedef typename LeafFunctor<LatticeSeed, ParamLeaf>::Type_t  LatticeSeedJIT;
  typedef typename REGType<typename SeedJIT::Subtype_t>::Type_t PSeedREG;

  SeedJIT ran_seed_jit(forEach(RNG::ran_seed, param_leaf, TreeCombine()));
  SeedJIT seed_tmp_jit(forEach(seed_tmp, param_leaf, TreeCombine()));
  // SeedJIT skewed_seed_jit(forEach(skewed_seed, param_leaf, TreeCombine()));
  SeedJIT ran_mult_n_jit(forEach(RNG::ran_mult_n, param_leaf, TreeCombine()));
  LatticeSeedJIT lattice_ran_mult_jit(forEach( *RNG::lattice_ran_mult , param_leaf, TreeCombine()));

  //  printme<View_t>();

  PSeedREG seed_reg;
  PSeedREG skewed_seed_reg;
  PSeedREG ran_mult_n_reg;
  PSeedREG lattice_ran_mult_reg;
  // typename SeedREG::Subtype_t seed_jit;
  // typename SeedREG::Subtype_t skewed_seed_jit;
  // typename SeedREG::Subtype_t ran_seed_jit_elem;
  // typename SeedREG::Subtype_t lattice_ran_mult_jit_elem;

  seed_reg.setup( ran_seed_jit.elem() );

  lattice_ran_mult_reg.setup( lattice_ran_mult_jit.elem( QDPTypeJITBase::Coalesced ) );

  skewed_seed_reg = seed_reg * lattice_ran_mult_reg;

  ran_mult_n_reg.setup( ran_mult_n_jit.elem() );

  fill_random( dest_jit.elem(QDPTypeJITBase::Coalesced) , seed_reg , skewed_seed_reg , ran_mult_n_reg );

  jit_value_t r_no_save = jit_ins_ne( r_idx , jit_val_create_const_int(0) );

  jit_label_t label_nosave;
  jit_ins_branch( function , label_nosave , r_no_save );
  seed_tmp_jit.elem() = seed_reg;
  jit_ins_label( function , label_nosave );

  if (Layout::primaryNode())
    function->write();
      
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad( &cuModule , fname );
  if (ret) QDP_error_exit("Error loading CUDA module '%s'",fname);

  ret = cuModuleGetFunction(&func, cuModule, "function");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}








template<class T, class T1, class RHS>
CUfunction
function_gather_build( void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  const char * fname = "ptx_gather.ptx";
  jit_function_t function = jit_create_function( fname );

  jit_value_t r_lo     = jit_add_param( function , jit_ptx_type::s32 );
  jit_value_t r_hi     = jit_add_param( function , jit_ptx_type::s32 );

  jit_value_t r_idx = jit_geom_get_linear_th_idx( function );  

  jit_value_t r_out_of_range       = jit_ins_ge( r_idx , r_hi );
  jit_ins_exit( function , r_out_of_range );

  jit_value_t r_perm_array_addr      = jit_add_param( function , jit_ptx_type::u64 );  // Site permutation array
  jit_value_t r_idx_mul_4            = jit_ins_mul( r_idx , jit_val_create_const_int(4) );
  jit_value_t r_perm_array_addr_load = jit_ins_add( r_perm_array_addr , r_idx_mul_4 );
  jit_value_t r_idx_perm             = jit_ins_load ( r_perm_array_addr_load , 0 , jit_ptx_type::s32 );

  jit_value_t r_gather_buffer        = jit_add_param( function , jit_ptx_type::u64 );  // Gather buffer

  //ParamLeaf param_leaf_idx( function , r_idx );
  ParamLeaf param_leaf_idx_perm( function , r_idx_perm );
  
  // ParamLeaf param_leaf_0( function , function.getRegIdx() , Jit::LatticeLayout::SCAL );
  // ParamLeaf param_leaf_soffset( function , param_leaf_0.getParamIndexFieldAndOption() , Jit::LatticeLayout::COAL );

  // Destination
  typedef typename JITType< OLattice<T> >::Type_t DestView_t;

  DestView_t dest_jit( function , r_gather_buffer , r_idx );

		       // param_leaf_0.getParamLattice( JITType<T>::Type_t::Size_t * WordSize<T>::Size ) ,
		       // Jit::LatticeLayout::SCAL );

  // typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  // FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view( forEach( rhs , param_leaf_idx_perm , TreeCombine() ) );

  //printme<View_t>();

  OpAssign()( dest_jit.elem( QDPTypeJITBase::Scalar ) , 
	      forEach(rhs_view, ViewLeaf( QDPTypeJITBase::Coalesced ) , OpCombine() ) );

  if (Layout::primaryNode())
    function->write();

  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, fname);
  if (ret) QDP_error_exit("Error loading CUDA module '%s'",fname);

  ret = cuModuleGetFunction(&func, cuModule, "function");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}


template<class T1, class RHS>
void
function_gather_exec( CUfunction function, void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
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

}



template<class T, class T1, class Op, class RHS>
void 
function_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  //  std::cout << "function_exec 0\n";

  ShiftPhase1 phase1;
  int offnode_maps = forEach(rhs, phase1 , BitOrCombine());
  //QDP_info("offnode_maps = %d",offnode_maps);

  void * idx_inner_dev = NULL;
  void * idx_face_dev = NULL;

  // lo <= idx < hi
  int lo = 0;
  int hi;
  int do_soffset_index;
  int faceCount;

  if (offnode_maps > 0) {
    int innerId, innerCount, faceId;
    innerId = MasterMap::Instance().getIdInner(offnode_maps);
    innerCount = MasterMap::Instance().getCountInner(offnode_maps);
    faceId = MasterMap::Instance().getIdFace(offnode_maps);
    faceCount = MasterMap::Instance().getCountFace(offnode_maps);
    idx_inner_dev = QDPCache::Instance().getDevicePtr( innerId );
    idx_face_dev = QDPCache::Instance().getDevicePtr( faceId );
    hi = innerCount;
    do_soffset_index = 1;
    //QDP_info("innerId = %d innerCount = %d faceId = %d  faceCount = %d",innerId,innerCount,faceId,faceCount);
  } else {
    hi = Layout::sitesOnNode();
    do_soffset_index = 0;
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

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

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

  if (!threadsPerBlock[hi-lo]) {
    // Auto tuning
    // Fist get a data field of the same size as "dest" 
    // where it's safe to do autotuning on.
    int tmpId = QDPCache::Instance().registrate( QDPCache::Instance().getSize( dest.getId() ) , 1 , NULL );
    void * devPtr = QDPCache::Instance().getDevicePtr( tmpId );
    //QDPCache::Instance().printLockSets();
    addr[addr_dest] = &devPtr;

    threadsPerBlock[hi-lo] = jit_autotuning(function,lo,hi,&addr[0]);

    // Restore original "dest" device address
    addr[addr_dest] = &addr_leaf.addr[0];
    QDPCache::Instance().signoff( tmpId );
    //QDPCache::Instance().printLockSets();

  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( hi-lo , threadsPerBlock[hi-lo] );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock[hi-lo],1,1,    0, 0, &addr[0] , 0);

  if (offnode_maps > 0) {
    ShiftPhase2 phase2;
    forEach(rhs, phase2 , NullCombine());

    hi = faceCount;
    idx_inner_dev = idx_face_dev;

    if (!threadsPerBlock[hi-lo]) {
      int tmpId = QDPCache::Instance().registrate( QDPCache::Instance().getSize( dest.getId() ) , 1 , NULL );
      void * devPtr = QDPCache::Instance().getDevicePtr( tmpId );
      addr[addr_dest] = &devPtr;
      threadsPerBlock[hi-lo] = jit_autotuning(function,lo,hi,&addr[0]);
      addr[addr_dest] = &addr_leaf.addr[0];
      QDPCache::Instance().signoff( tmpId );
    }
    now = getGeom( hi-lo , threadsPerBlock[hi-lo] );                                  
    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock[hi-lo],1,1,    0, 0, &addr[0] , 0);
  }
}

template<class T, class T1, class Op, class RHS>
void 
function_lat_sca_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  AddOpAddress<Op,AddressLeaf>::apply(op,addr_leaf);
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = 0;
  int hi = Layout::sitesOnNode();
  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
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

    threadsPerBlock = jit_autotuning(function,lo,hi,&addr[0]);

    // Restore original "dest" device address
    addr[addr_dest] = &addr_leaf.addr[0];
    QDPCache::Instance().signoff( tmpId );
    //QDPCache::Instance().printLockSets();

  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( hi-lo , threadsPerBlock );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

}






template<class T>
void 
function_zero_rep_exec(CUfunction function, OLattice<T>& dest, const Subset& s )
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = 0;
  int hi = Layout::sitesOnNode();
  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

#ifdef JIT_DO_MEMBER
  addr.push_back( &subset_member );
  //std::cout << "addr subset_member = " << addr[3] << " " << subset_member << "\n";
#endif

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    // Auto tuning
    threadsPerBlock = jit_autotuning(function,lo,hi,&addr[0]);
  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( hi-lo , threadsPerBlock );
  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);
}






#if 0



template<class T, class T1, class Op, class RHS>
void 
function_sca_sca_exec(CUfunction function, OScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  AddressLeaf addr_leaf;

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = 0;
  int hi = 1;

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

  kernel_geom_t now = getGeom( hi-lo , 1 );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    1,1,1,    0, 0, &addr[0] , 0);
}
#endif



template<class T>
void 
function_random_exec(CUfunction function, OLattice<T>& dest, const Subset& s , Seed& seed_tmp)
{
#if 1
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

  int junk_1 = forEach(RNG::ran_seed, addr_leaf, NullCombine());
  int junk_2 = forEach(seed_tmp, addr_leaf, NullCombine());
  int junk_3 = forEach(RNG::ran_mult_n, addr_leaf, NullCombine());
  int junk_4 = forEach(*RNG::lattice_ran_mult, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = 0;
  int hi = Layout::sitesOnNode();
  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  addr.push_back( &subset_member );
  //std::cout << "addr subset_member = " << addr[3] << " " << subset_member << "\n";

  int addr_dest=addr.size();
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr = " << addr_leaf.addr[i] << "\n";
  }

  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    // Auto tuning
    threadsPerBlock = jit_autotuning(function,lo,hi,&addr[0]);
  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  //QDP_info("Launching kernel with %d threads",hi-lo);

  kernel_geom_t now = getGeom( hi-lo , threadsPerBlock );

  CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

#endif
}


}

#endif
