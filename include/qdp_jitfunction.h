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

  ParamLeaf param_leaf(function);

  // Destination
  typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
  FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));

  // Now the arguments for the rhs
  typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, ParamLeaf, TreeCombine>::Type_t View_t;
  View_t rhs_view(forEach(rhs, param_leaf, TreeCombine()));

  // Automatically build the function
  // This is where the site loop would go. This version completely unrolls it.
  // Instead, want a jit generated loop

  // The site loop is implemented as CUDA thread parallelization

  //printme(rhs_view);

  op(dest_jit.elem(0), forEach(rhs_view, ViewLeaf(0), OpCombine()));

  function.write();

  CUresult ret;
  CUmodule cuModule;
  ret = cuModuleLoad(&cuModule, "ptxtest.ptx");
  if (ret) { std::cout << "Error loading CUDA module\n"; exit(1); }

  ret = cuModuleGetFunction(&func, cuModule, "func");
  if (ret) { std::cout << "Error getting function\n"; exit(1); }

  std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

  return func;
}



template<class T, class T1, class Op, class RHS>
void 
function_exec(CUfunction function, OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

  AddressLeaf addr_leaf;

  int junk_dst = forEach(dest, addr_leaf, NullCombine());
  int junk_rhs = forEach(rhs, addr_leaf, NullCombine());

  //QDPCache::Instance().printLockSets();

  std::vector<void*> addr;
  for(int i=0; i < addr_leaf.addr.size(); ++i) {
    addr.push_back( &addr_leaf.addr[i] );
    //std::cout << "addr=" << addr_leaf.addr[i] << "\n";
  }

  static int threadsPerBlock = 0;

  if (!threadsPerBlock) {
    // Auto tuning

    // Fist get a data field of the same size as "dest" where we can play on
    // (in case the final operator is an OpAddAssign, etc.)
    int tmpId = QDPCache::Instance().registrate( QDPCache::Instance().getSize( dest.getId() ) , 1 , NULL );
    void * devPtr = QDPCache::Instance().getDevicePtr( tmpId );
    //QDPCache::Instance().printLockSets();
    addr[0] = &devPtr;

    double best_time;
    int best_cfg=-1;
    bool first=true;
    for ( int cfg = 1 ; cfg <= DeviceParams::Instance().getMaxBlockX(); cfg *= 2 ) {
      kernel_geom_t now = getGeom( Layout::sitesOnNode() , cfg );

      StopWatch w;
      CUresult result = CUDA_SUCCESS;

      for (int i=0 ; i < 10 && result == CUDA_SUCCESS; i++) {
	if (i==1) w.start();
	result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    cfg,1,1,    0, 0, &addr[0] , 0);
	CudaDeviceSynchronize();
      }

      if (result == CUDA_SUCCESS) {
	w.stop();
	double time = w.getTimeInMicroseconds();
	QDP_info_primary("launched threads per block = %d grid = (%d,%d) (time=%f micro secs)",cfg,now.Nblock_x,now.Nblock_y,time);
	if (first) {
	  best_time = time;
	  best_cfg = cfg;
	  first = false;
	} else {
	  if (time < best_time) {
	    best_time = time;
	    best_cfg = cfg;
	  }
	}
      } else {
	QDP_info_primary("tried threads per block = %d, failed, code = %d ",cfg,result);
      }
    }

    if (best_cfg < 0)
      QDP_error_exit("Auto-tuning failed!");

    QDP_info_primary("Threads per block favored = %d  (time=%f micro secs)",best_cfg,best_time);
    threadsPerBlock = best_cfg;

    // Restore original "dest" device address
    addr[0] = &addr_leaf.addr[0];
    QDPCache::Instance().signoff( tmpId );
    //QDPCache::Instance().printLockSets();

  } else {
    //QDP_info_primary("Previous auto-tuning result = %d",threadsPerBlock);
  }

  kernel_geom_t now = getGeom( Layout::sitesOnNode() , threadsPerBlock );
  CUresult result = CUDA_SUCCESS;
  result = cuLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);

  if (DeviceParams::Instance().getSyncDevice()) {  
    QDP_info_primary("Pulling the brakes: device sync after kernel launch!");
    CudaDeviceSynchronize();
  }

}


}

#endif
