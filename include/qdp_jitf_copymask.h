#ifndef QDP_JITF_COPYMASK_H
#define QDP_JITF_COPYMASK_H

#include "qmp.h"

namespace QDP {

  template<class T,class T1>
  CUfunction
  function_copymask_build( OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
  {
    CUfunction func;

    std::string fname("ptx_copymask.ptx");
    Jit function(fname.c_str(),"func");

    ParamLeaf param_leaf(function,function.getRegIdx() , Jit::LatticeLayout::COAL );

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    typedef typename LeafFunctor<OLattice<T1>, ParamLeaf>::Type_t  FuncRet1_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));
    FuncRet1_t mask_jit(forEach(mask, param_leaf, TreeCombine()));

    copymask( dest_jit.elem(0) , mask_jit.elem(0) , src_jit.elem(0) );

    if (Layout::primaryNode())
      function.write();
      
    QMP_barrier();

    CUresult ret;
    CUmodule cuModule;
    ret = cuModuleLoad(&cuModule, fname.c_str());
    if (ret) QDP_error_exit("Error loading CUDA module '%s'",fname.c_str());

    ret = cuModuleGetFunction(&func, cuModule, "func");
    if (ret) { std::cout << "Error getting function\n"; exit(1); }

    return func;
  }



  template<class T,class T1>
  void 
  function_copymask_exec(CUfunction function, OLattice<T>& dest, const OLattice<T1>& mask, const OLattice<T>& src )
  {
    AddressLeaf addr_leaf;

    int junk_0 = forEach(dest, addr_leaf, NullCombine());
    int junk_1 = forEach(src, addr_leaf, NullCombine());
    int junk_2 = forEach(mask, addr_leaf, NullCombine());

    // lo <= idx < hi
    int lo = 0;
    int hi = Layout::sitesOnNode();

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
  }

}
#endif
