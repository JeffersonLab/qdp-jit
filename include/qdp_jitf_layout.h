#ifndef QDP_JITF_LAYOUT_H
#define QDP_JITF_LAYOUT_H

#include "qmp.h"

namespace QDP {

  template<class T>
  void *
  function_layout_to_jit_build( const OLattice<T>& dest )
  {
    JitMainLoop loop;

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(dest, param_leaf, TreeCombine()));

    typedef typename REGType<typename FuncRet_t::Subtype_t>::Type_t REGFuncRet_t;

    IndexDomainVector r_idx = loop.getIdx();

    REGFuncRet_t src_reg;
    src_reg.setup ( src_jit.elem( JitDeviceLayout::Scalar , r_idx ) );

    dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ) = src_reg;

    loop.done();

    return jit_function_epilogue_get("jit_layout.ptx");
  }



  template<class T>
  void 
  function_layout_to_jit_exec(void * function, T *dest, T *src )
  {
    AddressLeaf addr_leaf;
    jit_get_empty_arguments(addr_leaf);

    addr_leaf.setAddr( dest );
    addr_leaf.setAddr( src );

    int th_count = Layout::sitesOnNode();

    std::cout << "calling layout(to JIT)..\n";

    jit_dispatch(function,th_count,addr_leaf);
  }










  template<class T>
  void *
  function_layout_to_native_build( const OLattice<T>& dest )
  {
    JitMainLoop loop;

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(dest, param_leaf, TreeCombine()));

    typedef typename REGType<typename FuncRet_t::Subtype_t>::Type_t REGFuncRet_t;

    IndexDomainVector r_idx = loop.getIdx();

    REGFuncRet_t src_reg;
    src_reg.setup ( src_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

    dest_jit.elem( JitDeviceLayout::Scalar , r_idx ) = src_reg;

    loop.done();

    return jit_function_epilogue_get("jit_layout.ptx");
  }



  template<class T>
  void 
  function_layout_to_native_exec(void * function, T *dest, T *src )
  {
    AddressLeaf addr_leaf;
    jit_get_empty_arguments(addr_leaf);

    addr_leaf.setAddr( dest );
    addr_leaf.setAddr( src );

    int th_count = Layout::sitesOnNode();

    std::cout << "calling layout(to native)..\n";

    jit_dispatch(function,th_count,addr_leaf);
  }

}
#endif
