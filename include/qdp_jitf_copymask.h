#ifndef QDP_JITF_COPYMASK_H
#define QDP_JITF_COPYMASK_H

#include "qmp.h"

namespace QDP {

  template<class T,class T1>
  CUfunction
  function_copymask_build( OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
  {
    CUfunction func;

    const char * fname = "ptx_copymask.ptx";
    jit_function_t function = jit_create_function( fname );


  jit_value_t r_lo           = jit_add_param( function , jit_ptx_type::s32 );
  jit_value_t r_hi           = jit_add_param( function , jit_ptx_type::s32 );
  jit_value_t r_idx          = jit_geom_get_linear_th_idx( function );  
  jit_value_t r_out_of_range = jit_ins_ge( r_idx , r_hi );
  jit_ins_exit( function , r_out_of_range );

  ParamLeaf param_leaf( function , r_idx );



    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    typedef typename LeafFunctor<OLattice<T1>, ParamLeaf>::Type_t  FuncRet1_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));
    FuncRet1_t mask_jit(forEach(mask, param_leaf, TreeCombine()));

    typedef typename REGType<typename FuncRet_t::Subtype_t>::Type_t REGFuncRet_t;
    typedef typename REGType<typename FuncRet1_t::Subtype_t>::Type_t REGFuncRet1_t;

    REGFuncRet_t src_reg;
    REGFuncRet1_t mask_reg;
    src_reg.setup ( src_jit.elem( QDPTypeJITBase::Coalesced ) );
    mask_reg.setup( mask_jit.elem( QDPTypeJITBase::Coalesced ) );

    copymask( dest_jit.elem( QDPTypeJITBase::Coalesced ) , mask_reg , src_reg );

    if (Layout::primaryNode())
      function->write();
      
    QMP_barrier();

    CUresult ret;
    CUmodule cuModule;
    ret = cuModuleLoad(&cuModule, fname);
    if (ret) QDP_error_exit("Error loading CUDA module '%s'",fname);

    ret = cuModuleGetFunction(&func, cuModule, "function");
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

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threadsPerBlock,1,1,    0, 0, &addr[0] , 0);
  }

}
#endif
