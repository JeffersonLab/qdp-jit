#ifndef QDP_JITF_COPYMASK_H
#define QDP_JITF_COPYMASK_H

#include "qmp.h"

namespace QDP {

  template<class T,class T1>
  void *
  function_copymask_build( OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
  {
    llvm_start_new_function();

    ParamRef p_start        = llvm_add_param<int>();
    ParamRef p_end          = llvm_add_param<int>();

    ParamLeaf param_leaf;

    typedef typename LeafFunctor<OLattice<T>, ParamLeaf>::Type_t  FuncRet_t;
    typedef typename LeafFunctor<OLattice<T1>, ParamLeaf>::Type_t  FuncRet1_t;

    FuncRet_t dest_jit(forEach(dest, param_leaf, TreeCombine()));
    FuncRet_t src_jit(forEach(src, param_leaf, TreeCombine()));
    FuncRet1_t mask_jit(forEach(mask, param_leaf, TreeCombine()));

    llvm::Value * r_start        = llvm_derefParam( p_start );
    llvm::Value * r_end          = llvm_derefParam( p_end );

    typedef typename REGType<typename FuncRet_t::Subtype_t>::Type_t REGFuncRet_t;
    typedef typename REGType<typename FuncRet1_t::Subtype_t>::Type_t REGFuncRet1_t;


  llvm::BasicBlock * block_entry_point = llvm_get_insert_point();
  llvm::BasicBlock * block_site_loop = llvm_new_basic_block();
  llvm::BasicBlock * block_site_loop_exit = llvm_new_basic_block();
  llvm::BasicBlock * block_end_copymask = llvm_new_basic_block();
  llvm_branch( block_site_loop );
  llvm_set_insert_point(block_site_loop);

  llvm::PHINode* r_idx      = llvm_phi( r_start->getType() , 2 );
  llvm::Value*   r_idx_new;


    REGFuncRet_t src_reg;
    REGFuncRet1_t mask_reg;
    src_reg.setup ( src_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );
    mask_reg.setup( mask_jit.elem( JitDeviceLayout::Coalesced , r_idx ) );

    copymask( dest_jit.elem( JitDeviceLayout::Coalesced , r_idx ) , mask_reg , src_reg );

  llvm_branch( block_end_copymask );
  llvm_set_insert_point(block_end_copymask);


  r_idx_new = llvm_add( r_idx , llvm_create_value(1) );

  r_idx->addIncoming( r_idx_new , block_end_copymask );
  r_idx->addIncoming( r_start , block_entry_point );

  llvm::Value * r_exit_cond = llvm_ge( r_idx_new , r_end );

  llvm_cond_branch( r_exit_cond , block_site_loop_exit , block_site_loop );

  llvm_set_insert_point(block_site_loop_exit);

  return jit_function_epilogue_get("jit_copymask.ptx");
  //return llvm_get_cufunction("jit_copymask.ptx");
  }



  template<class T,class T1>
  void 
  function_copymask_exec(void * function, OLattice<T>& dest, const OLattice<T1>& mask, const OLattice<T>& src )
  {
    AddressLeaf addr_leaf;

  addr_leaf.addr.push_back(AddressLeaf::Types(0));
  addr_leaf.addr.push_back(AddressLeaf::Types(Layout::sitesOnNode()));

    int junk_0 = forEach(dest, addr_leaf, NullCombine());
    int junk_1 = forEach(src, addr_leaf, NullCombine());
    int junk_2 = forEach(mask, addr_leaf, NullCombine());

  void (*FP)(void*) = (void (*)(void*))(intptr_t)function;

  std::cout << "calling copymask(Lattice)..\n";
  FP( addr_leaf.addr.data() );
  std::cout << "..done\n";

  }

}
#endif
