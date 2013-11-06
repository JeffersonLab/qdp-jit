#ifndef QDP_JITF_COPYMASK_H
#define QDP_JITF_COPYMASK_H

#include "qmp.h"

namespace QDP {

  template<class T,class T1>
  void *
  function_copymask_build( OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
  {
    JitMainLoop loop;

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    typedef typename LeafFunctor<OLattice<T1>, ParamLeaf>::Type_t  FuncRet1_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));
    FuncRet1_t mask_jit(forEach(mask, param_leaf, TreeCombine()));

    typedef typename REGType<typename FuncRet_t::Subtype_t>::Type_t REGFuncRet_t;
    typedef typename REGType<typename FuncRet1_t::Subtype_t>::Type_t REGFuncRet1_t;

    //llvm::Value * r_idx = loop.getIdx();
    IndexDomainVector idx = loop.getIdx();

    REGFuncRet_t src_reg;
    REGFuncRet1_t mask_reg;
    src_reg.setup ( src_jit.elem( JitDeviceLayout::Coalesced , idx ) );
    mask_reg.setup( mask_jit.elem( JitDeviceLayout::Coalesced , idx ) );

    copymask( dest_jit.elem( JitDeviceLayout::Coalesced , idx ) , mask_reg , src_reg );

    loop.done();

    return jit_function_epilogue_get("jit_copymask.ptx");
  }



  template<class T,class T1>
  void 
  function_copymask_exec(void * function, OLattice<T>& dest, const OLattice<T1>& mask, const OLattice<T>& src )
  {
    AddressLeaf addr_leaf;
    jit_get_empty_arguments(addr_leaf);

    int junk_0 = forEach(dest, addr_leaf, NullCombine());
    int junk_1 = forEach(src, addr_leaf, NullCombine());
    int junk_2 = forEach(mask, addr_leaf, NullCombine());

    int th_count = Layout::sitesOnNode();

    std::cout << "calling copymask(Lattice)..\n";
    jit_call(function,th_count,addr_leaf);
  }

}
#endif
