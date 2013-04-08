#ifndef QDP_JITF_COPYMASK_H
#define QDP_JITF_COPYMASK_H

#include "qmp.h"

namespace QDP {

  template<class T,class T1>
  CUfunction
  function_copymask_build( OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
  {
    CUfunction func;

    jit_start_new_function();

    jit_value r_lo           = jit_add_param( jit_ptx_type::s32 );
    jit_value r_hi           = jit_add_param( jit_ptx_type::s32 );
    jit_value r_idx          = jit_geom_get_linear_th_idx();  
    jit_value r_out_of_range = jit_ins_ge( r_idx , r_hi );
    jit_ins_exit( r_out_of_range );

    ParamLeaf param_leaf( r_idx );

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    typedef typename LeafFunctor<OLattice<T1>, ParamLeaf>::Type_t  FuncRet1_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));
    FuncRet1_t mask_jit(forEach(mask, param_leaf, TreeCombine()));

    typedef typename REGType<typename FuncRet_t::Subtype_t>::Type_t REGFuncRet_t;
    typedef typename REGType<typename FuncRet1_t::Subtype_t>::Type_t REGFuncRet1_t;

    REGFuncRet_t src_reg;
    REGFuncRet1_t mask_reg;
    src_reg.setup ( src_jit.elem( JitDeviceLayout::Coalesced ) );
    mask_reg.setup( mask_jit.elem( JitDeviceLayout::Coalesced ) );

    copymask( dest_jit.elem( JitDeviceLayout::Coalesced ) , mask_reg , src_reg );

    return jit_get_cufunction("ptx_copymask.ptx");
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

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);
  }

}
#endif
