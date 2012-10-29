#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

#include "qmp.h"

namespace QDP {
  template<class T1,class T2>
  CUfunction 
  function_sum_build()
  {
    std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    std::string fname("ptxsum.ptx");
    Jit function(fname.c_str(),"func");

    OLatticeJIT<typename JITContainerType<T2>::Type_t> sdata( function , 
							      function.addSharedMemLatticeBaseAddr( function.getTID() , 
												    JITContainerType<T2>::Type_t::Size_t * WordSize<T2>::Size ),
							      Jit::LatticeLayout::SCAL );

    //sdata.elem(0) += sdata.elem(0);

    

    int r_s = function.getRegs( Jit::s32 , 1 );
    int r_pred_s = function.getRegs( Jit::pred , 1 );

    int r_zero = function.getRegs( Jit::s32 , 1 );
    function.asm_mov_literal( r_zero , (int)0 );

    int r_one = function.getRegs( Jit::u32 , 1 );
    function.asm_mov_literal( r_one , (unsigned)1 );


    function.insert_label("LOOP_S");

    function.asm_cmp( Jit::CmpOp::le , r_pred_s , r_s , r_zero );

    function.addCondBranchPredToLabel( r_pred_s , "LOOP_S_END" );


    int r_s_p_tid = function.getRegs( Jit::s32 , 1 );
    function.asm_add( r_s_p_tid , function.getTID() , r_s );


    int r_s_p_tid_u32 = function.getRegs( Jit::u32 , 1 );
    function.asm_cvt( r_s_p_tid_u32 , r_s_p_tid );
    int r_multiplier_u32 = function.getRegs( Jit::u32 , 1 );
    function.asm_mov_literal( r_multiplier_u32 , (unsigned)JITContainerType<T2>::Type_t::Size_t * WordSize<T2>::Size );
    int r_offset_u64 = function.getRegs( Jit::u64 , 1 );
    function.asm_mul( r_offset_u64 , r_s_p_tid_u32 , r_multiplier_u32 );
    int r_addr_u64 = function.getRegs( Jit::u64 , 1 );
    function.asm_add( r_addr_u64 , function.getSDATA() , r_offset_u64 );
    function.set_state_space( r_addr_u64 , Jit::SHARED );

    typename JITContainerType<T2>::Type_t sdata_p( curry_t(function,r_addr_u64,1,0) );

    sdata.elem(0) += sdata_p;
    //sdata.elem(0) = sdata.elem(0) + sdata.elem(0);

    function.asm_shr( r_s , r_s , r_one );
    function.addBranchToLabel( "LOOP_S" );
    function.insert_label("LOOP_S_END");

    if (Layout::primaryNode())
      function.write();
      
    QMP_barrier();

    CUresult ret;
    CUmodule cuModule;
    ret = cuModuleLoad(&cuModule, fname.c_str());
    if (ret) QDP_error_exit("Error loading CUDA module '%s'",fname.c_str());

    ret = cuModuleGetFunction(&func, cuModule, "func");
    if (ret) { std::cout << "Error getting function\n"; exit(1); }

    std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

    return func;
  }


}

#endif
