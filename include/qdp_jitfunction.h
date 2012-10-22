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

  std::cout << "function = " << (void*)&function <<"\n";

  ParamLeaf param_leaf(function,function.getRegIdx() , Jit::LatticeLayout::COAL );
  ParamLeaf param_leaf_indexed( function , param_leaf.getParamIndexField() , Jit::LatticeLayout::COAL);

  // Destination
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf_indexed, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf_indexed, TreeCombine()));

  printme<View_t>();

  op(dest_jit.elem( Jit::LatticeLayout::COAL , 0 ), forEach(rhs_view, ViewLeaf( Jit::LatticeLayout::COAL , 0 ), OpCombine()));

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

  std::cout << "function = " << (void*)&function <<"\n";

  ParamLeaf param_leaf_0( function , function.getRegIdx() , Jit::LatticeLayout::SCAL );
  ParamLeaf param_leaf_soffset( function , param_leaf_0.getParamIndexField() , Jit::LatticeLayout::COAL );

  // Destination
  typedef typename JITContainerType< OLattice<T> >::Type_t DestView_t;
  QDP_info("------------ %d",JITContainerType<T>::Type_t::Size_t * WordSize<T>::Size);
  DestView_t dest_jit( function , param_leaf_0.getParamLattice( JITContainerType<T>::Type_t::Size_t * WordSize<T>::Size ) );

  // typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  // FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view( forEach( rhs , param_leaf_soffset , TreeCombine() ) );

  printme<View_t>();

  OpAssign()( dest_jit.elem( Jit::LatticeLayout::SCAL , 0 ) , forEach(rhs_view, ViewLeaf( Jit::LatticeLayout::COAL , 0 ) , OpCombine() ) );

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
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  QDPCache::Instance().printLockSets();

  // lo <= idx < hi
  int lo = 0;
  int hi = map.soffset().size();
  QDP_info("gather sites into send_buf lo=%d hi=%d",lo,hi);

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

  QDPCache::Instance().printLockSets();

  std::vector<void*> addr;

  addr.push_back( &lo );
  std::cout << "addr lo =" << addr[0] << "\n";

  addr.push_back( &hi );
  std::cout << "addr hi =" << addr[1] << "\n";

  addr.push_back( &soffsetsDev );
  std::cout << "addr goffsetsDev =" << addr[2] << " " << soffsetsDev << "\n";

  addr.push_back( &send_buf );
  std::cout << "addr send_buf =" << addr[3] << " " << send_buf << "\n";

  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    std::cout << "addr rhs =" << addr[addr.size()-1] << " " << addr_leaf.addr[i] << "\n";
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
  // void * subset_member = QDPCache::Instance().getDevicePtr( s.getIdMemberTable() );

  AddressLeaf addr_leaf;

  int junk_dest = forEach(dest, addr_leaf, NullCombine());
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  // lo <= idx < hi
  int lo = 0;
  int hi = innerCount;

  std::vector<void*> addr;

  addr.push_back( &lo );
  //std::cout << "addr lo = " << addr[0] << " lo=" << lo << "\n";

  addr.push_back( &hi );
  //std::cout << "addr hi = " << addr[1] << " hi=" << hi << "\n";

  addr.push_back( &idx_inner_dev );
  //std::cout << "addr idx_inner_dev = " << addr[2] << " " << idx_inner_dev << "\n";

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


}

#endif
