#ifndef QDP_JITF_COPYMASK_H
#define QDP_JITF_COPYMASK_H

#include "qmp.h"

namespace QDP {

  template<class T,class T1>
  CUfunction
  function_copymask_build( OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
  {
    if (ptx_db::db_enabled) {
      CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (func)
	return func;
    }

    llvm_start_new_function();

    llvm_add_param<int>();   // we don't need p_lo, since copymask on sublattices is not jitted
    ParamRef p_hi          = llvm_add_param<int>();

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    typedef typename LeafFunctor<OLattice<T1>, ParamLeaf>::Type_t  FuncRet1_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));
    FuncRet1_t mask_jit(forEach(mask, param_leaf, TreeCombine()));

    //llvm::Value * r_lo      = llvm_derefParam( p_lo );
    llvm::Value * r_hi      = llvm_derefParam( p_hi );

    llvm::Value* r_idx          = llvm_thread_idx();  
    llvm::Value* r_out_of_range = llvm_ge( r_idx , r_hi );
    llvm_cond_exit( r_out_of_range );

    typedef typename REGType<typename FuncRet_t::Subtype_t>::Type_t REGFuncRet_t;
    typedef typename REGType<typename FuncRet1_t::Subtype_t>::Type_t REGFuncRet1_t;

    REGFuncRet_t src_reg;
    REGFuncRet1_t mask_reg;
    src_reg.setup ( src_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );
    mask_reg.setup( mask_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

    copymask( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ) , mask_reg , src_reg );

    llvm_exit();

    return llvm_get_cufunction("jit_copymask.ptx", __PRETTY_FUNCTION__ );
  }



  template<class T,class T1>
  void 
  function_copymask_exec(CUfunction function, OLattice<T>& dest, const OLattice<T1>& mask, const OLattice<T>& src )
  {
    AddressLeaf addr_leaf(all);

    forEach(dest, addr_leaf, NullCombine());
    forEach(src, addr_leaf, NullCombine());
    forEach(mask, addr_leaf, NullCombine());

    // lo <= idx < hi
    int lo = 0;
    int hi = Layout::sitesOnNode();

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

    jit_launch(function,hi-lo,addr);
  }

}
#endif
