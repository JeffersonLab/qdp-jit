#ifndef QDP_JITFUNC_H
#define QDP_JITFUNC_H

#include "qmp.h"

namespace QDP {


template<class T, class T1, class Op, class RHS>
CUfunction
function_build(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  Jit function("ptxtest.ptx","func");

  //std::cout << "function = " << (void*)&function <<"\n";

  ParamLeaf param_leaf(function,function.getRegIdx() , Jit::LatticeLayout::COAL );
  ParamLeaf param_leaf_indexed( function , param_leaf.getParamIndexFieldAndOption() , Jit::LatticeLayout::COAL);
  function.addParamMemberArray( param_leaf.r_idx );

  // Destination
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf_indexed, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf_indexed, TreeCombine()));

  printme<View_t>();

  op(dest_jit.elem( 0 ), forEach(rhs_view, ViewLeaf( 0 ), OpCombine()));

  if (Layout::primaryNode())
    function.write();
      
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, "ptxtest.ptx");
  if (ret) { std::cout << "Error loading CUDA module\n"; exit(1); }

  ret = cuModuleGetFunction(&func, cuModule, "func");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}


template<class T>
CUfunction
function_random_build(OLattice<T>& dest, LatticeSeed& seed, LatticeSeed& skewed_seed)
{
#if 1
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  Jit function("ptxrandom.ptx","func");

  ParamLeaf param_leaf(function,function.getRegIdx() , Jit::LatticeLayout::COAL );
  function.addParamMemberArray( param_leaf.r_idx );

  // Destination
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // RNG::ran_seed
  //typedef typename LeafFunctor<Seed, ParamLeaf>::Type_t  SeedJIT;
  typedef typename LeafFunctor<LatticeSeed, ParamLeaf>::Type_t  LatticeSeedJIT;

  LatticeSeedJIT ran_seed_jit(forEach(*RNG::lat_ran_seed, param_leaf, TreeCombine()));
  LatticeSeedJIT seed_jit(forEach(seed, param_leaf, TreeCombine()));
  LatticeSeedJIT skewed_seed_jit(forEach(skewed_seed, param_leaf, TreeCombine()));
  LatticeSeedJIT ran_mult_n_jit(forEach(*RNG::lat_ran_mult_n, param_leaf, TreeCombine()));
  LatticeSeedJIT lattice_ran_mult_jit(forEach( *RNG::lattice_ran_mult , param_leaf, TreeCombine()));

  //  printme<View_t>();

  seed_jit.elem(0)        = ran_seed_jit.elem(0);
  skewed_seed_jit.elem(0) = ran_seed_jit.elem(0) * lattice_ran_mult_jit.elem(0);

  fill_random( dest_jit.elem(0) , seed_jit , skewed_seed_jit , ran_mult_n_jit );

  if (Layout::primaryNode())
    function.write();
      
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, "ptxrandom.ptx");
  if (ret) { std::cout << "Error loading CUDA module\n"; exit(1); }

  ret = cuModuleGetFunction(&func, cuModule, "func");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
#endif
}



template<class T>
CUfunction
function_zero_rep_build(OLattice<T>& dest)
{
#if 1
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  CUfunction func;

  Jit function("ptx_zero_rep.ptx","func");

  ParamLeaf param_leaf(function,function.getRegIdx() , Jit::LatticeLayout::COAL );
  function.addParamMemberArray( param_leaf.r_idx );

  // Destination
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  zero_rep( dest_jit.elem(0) );

  if (Layout::primaryNode())
    function.write();
      
  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, "ptx_zero_rep.ptx");
  if (ret) { std::cout << "Error loading CUDA module\n"; exit(1); }

  ret = cuModuleGetFunction(&func, cuModule, "func");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
#endif
}



template<class T, class T1, class RHS>
CUfunction
function_gather_build( void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  // void * soffsetDev = QDPCache::Instance().getDevicePtr( map.getSoffsetsId() );
  // QDP_info("soffsetDev = %p",soffsetDev);

  // ShiftPhase1 phase1;
  // int maps_involved = forEach(rhs, phase1 , BitOrCombine());

  // QDP_info("maps_involved=%d",maps_involved);

  // if (maps_involved > 0) {
  //   int innerId = MasterMap::Instance().getIdInner(maps_involved);
  //   int innerCount = MasterMap::Instance().getCountInner(maps_involved);
  //   int faceId = MasterMap::Instance().getIdFace(maps_involved);
  //   int faceCount = MasterMap::Instance().getCountFace(maps_involved);
  // }

  CUfunction func;

  Jit function("ptxgather.ptx","func");

  //std::cout << "function = " << (void*)&function <<"\n";
  
  ParamLeaf param_leaf_0( function , function.getRegIdx() , Jit::LatticeLayout::SCAL );
  ParamLeaf param_leaf_soffset( function , param_leaf_0.getParamIndexFieldAndOption() , Jit::LatticeLayout::COAL );

  // Destination
  typedef typename JITContainerType< OLattice<T> >::Type_t DestView_t;
  //QDP_info("------------ %d",JITContainerType<T>::Type_t::Size_t * WordSize<T>::Size);
  DestView_t dest_jit( function , 
		       param_leaf_0.getParamLattice( JITContainerType<T>::Type_t::Size_t * WordSize<T>::Size ) ,
		       Jit::LatticeLayout::SCAL );

  // typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  // FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view( forEach( rhs , param_leaf_soffset , TreeCombine() ) );

  //printme<View_t>();

  OpAssign()( dest_jit.elem( 0 ) , forEach(rhs_view, ViewLeaf( 0 ) , OpCombine() ) );

#if 1
  if (Layout::primaryNode())
    function.write();
#endif

  QMP_barrier();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, "ptxgather.ptx");
  if (ret) { std::cout << "Error loading CUDA module\n"; exit(1); }

  ret = cuModuleGetFunction(&func, cuModule, "func");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}


template<class T1, class RHS>
void
function_gather_exec( CUfunction function, void* send_buf , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs )
{
#if 1
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  //QDPCache::Instance().printLockSets();

  // lo <= idx < hi
  int lo = 0;
  int hi = map.soffset().size();
  int do_soffset_index = 1;

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

  addr.push_back( &do_soffset_index );
  //std::cout << "addr do_soffset_index =" << addr[2] << " " << do_soffset_index << "\n";

  addr.push_back( &soffsetsDev );
  //std::cout << "addr soffsetsDev =" << addr[3] << " " << soffsetsDev << "\n";

  addr.push_back( &send_buf );
  //std::cout << "addr send_buf =" << addr[4] << " " << send_buf << "\n";

  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr rhs =" << addr[addr.size()-1] << " " << addr_leaf.addr[i] << "\n";
  }



  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    threadsPerBlock = jit_autotuning(function,lo,hi,&addr[0]);
  } else {
    //QDP_info_primary("Previous gather_function auto-tuning result = %d",threadsPerBlock);
  }

  kernel_geom_t now = getGeom( hi-lo , threadsPerBlock );
  CUresult result = CUDA_SUCCESS;
  result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

  if (DeviceParams::Instance().getSyncDevice()) {  
    QDP_info_primary("Pulling the brakes: device sync after kernel launch!");
    CudaDeviceSynchronize();
  }
#endif
}




template<class T, class T1, class Op, class RHS>
void 
function_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  ShiftPhase1 phase1;
  int offnode_maps = forEach(rhs, phase1 , BitOrCombine());
  //QDP_info("offnode_maps = %d",offnode_maps);

  int innerId = MasterMap::Instance().getIdInner(offnode_maps);
  int innerCount = MasterMap::Instance().getCountInner(offnode_maps);
  int faceId = MasterMap::Instance().getIdFace(offnode_maps);
  int faceCount = MasterMap::Instance().getCountFace(offnode_maps);

  void * idx_inner_dev = QDPCache::Instance().getDevicePtr( innerId );
  void * idx_face_dev = QDPCache::Instance().getDevicePtr( faceId );
  void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

  AddressLeaf addr_leaf;

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = 0;
  int hi = innerCount;
  int do_soffset_index = (int)(offnode_maps > 0);

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
  CUresult result = CUDA_SUCCESS;
  result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

  if (DeviceParams::Instance().getSyncDevice()) {  
    QDP_info_primary("Pulling the brakes: device sync after kernel launch!");
    CudaDeviceSynchronize();
  }

  if (offnode_maps > 0) {
    //QDP_info_primary("PHASE2");
    ShiftPhase2 phase2;
    forEach(rhs, phase2 , NullCombine());

    hi = faceCount;
    idx_inner_dev = idx_face_dev;

    //QDP_info_primary("PHASE2 launch");
    now = getGeom( hi-lo , threadsPerBlock );
    result = CUDA_SUCCESS;
    result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

    if (DeviceParams::Instance().getSyncDevice()) {  
      QDP_info_primary("Pulling the brakes: device sync after kernel launch!");
      CudaDeviceSynchronize();
    }
  }


}



template<class T>
void 
function_random_exec(CUfunction function, OLattice<T>& dest, const Subset& s, LatticeSeed& seed, LatticeSeed& skewed_seed )
{
#if 1
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_0 = forEach(dest, addr_leaf, NullCombine());

  int junk_1 = forEach(*RNG::lat_ran_seed, addr_leaf, NullCombine());
  int junk_2 = forEach(seed, addr_leaf, NullCombine());
  int junk_3 = forEach(skewed_seed, addr_leaf, NullCombine());
  int junk_4 = forEach(*RNG::lat_ran_mult_n, addr_leaf, NullCombine());
  int junk_5 = forEach(*RNG::lattice_ran_mult, addr_leaf, NullCombine());

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
  CUresult result = CUDA_SUCCESS;
  result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

  if (DeviceParams::Instance().getSyncDevice()) {  
    QDP_info_primary("Pulling the brakes: device sync after kernel launch!");
    CudaDeviceSynchronize();
  }

#endif
}



template<class T>
void 
function_zero_rep_exec(CUfunction function, OLattice<T>& dest, const Subset& s )
{
#if 1
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
  CUresult result = CUDA_SUCCESS;
  result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

  if (DeviceParams::Instance().getSyncDevice()) {  
    QDP_info_primary("Pulling the brakes: device sync after kernel launch!");
    CudaDeviceSynchronize();
  }

#endif
}


}

#endif
