#ifndef QDP_JITF_GLOBMAX_H
#define QDP_JITF_GLOBMAX_H

// this code is not used


#include "qmp.h"

namespace QDP {


void function_global_max_exec( CUfunction function, 
			       int size, int threads, int blocks, int shared_mem_usage,
			       void *d_idata, void *d_odata);


  template<class T2, Jit::LatticeLayout InLayout >
  CUfunction 
  function_global_max_build()
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    std::string fname("ptx_globalmax.ptx");
    Jit function(fname.c_str(),"func");

    function.addParamIndexFieldAndOption();
    
    OLatticeJIT<typename JITType<T2>::Type_t> idata( function , 
							      function.addParamLatticeBaseAddr( function.getRegIdx() , 
												JITType<T2>::Type_t::Size_t * WordSize<T2>::Size ),
							      InLayout );
    OLatticeJIT<typename JITType<T2>::Type_t> odata( function , 
							      function.addParamLatticeBaseAddr( function.getRegBlockIdx() , 
												JITType<T2>::Type_t::Size_t * WordSize<T2>::Size ),
							      Jit::LatticeLayout::SCAL );


    OLatticeJIT<typename JITType<T2>::Type_t> sdata( function , 
							      function.addSharedMemLatticeBaseAddr( function.getTID() , 
												    JITType<T2>::Type_t::Size_t * WordSize<T2>::Size ),
							      Jit::LatticeLayout::SCAL );

    //sdata.elem(0) += sdata.elem(0);

    int r_pred_idx = function.getRegs( Jit::pred , 1 );
    function.asm_cmp( Jit::CmpOp::ge , r_pred_idx , function.getRegIdxNoIndex() , function.getRegHi() );
    function.addCondBranchPredToLabel( r_pred_idx , "ZERO_REP" );

    sdata.elem(0) = idata.elem(0);

    function.addBranchToLabel( "ZERO_REP_END" );
    function.insert_label("ZERO_REP");

    zero_rep( sdata.elem(0) );

    function.insert_label("ZERO_REP_END");
    function.asm_bar_sync(1);

    int r_one = function.getRegs( Jit::u32 , 1 );
    function.asm_mov_literal( r_one , (unsigned)1 );
    int r_s = function.getRegs( Jit::s32 , 1 );
    int r_pred_s = function.getRegs( Jit::pred , 1 );
    int r_pred_tid = function.getRegs( Jit::pred , 1 );
    int r_pred_block = function.getRegs( Jit::pred , 1 );
    int r_pred_pow = function.getRegs( Jit::pred , 1 );

    int r_zero = function.getRegs( Jit::s32 , 1 );
    function.asm_mov_literal( r_zero , (int)0 );

    function.asm_mov_literal( r_s , (int)1 );

    // Next power of 2 of blockDimX
    function.insert_label("POWER_START");
    function.asm_cmp( Jit::CmpOp::ge , r_pred_pow , r_s , function.getBlockDimX() );
    function.addCondBranchPredToLabel( r_pred_pow , "POWER_END" );
    function.asm_shl( r_s , r_s , r_one );
    function.addBranchToLabel( "POWER_START" );
    function.insert_label("POWER_END");

    function.asm_shr( r_s , r_s , r_one );

    function.insert_label("LOOP_S");

    function.asm_cmp( Jit::CmpOp::le , r_pred_s , r_s , r_zero );

    function.addCondBranchPredToLabel( r_pred_s , "LOOP_S_END" );

    function.asm_cmp( Jit::CmpOp::ge , r_pred_tid , function.getTID() , r_s );

    function.addCondBranchPredToLabel( r_pred_tid , "SYNC" );

    int r_s_p_tid = function.getRegs( Jit::s32 , 1 );
    function.asm_add( r_s_p_tid , function.getTID() , r_s );

    function.asm_cmp( Jit::CmpOp::ge , r_pred_block , r_s_p_tid , function.getBlockDimX() );

    function.addCondBranchPredToLabel( r_pred_block , "SYNC" );

    int r_s_p_tid_u32 = function.getRegs( Jit::u32 , 1 );
    function.asm_cvt( r_s_p_tid_u32 , r_s_p_tid );
    int r_multiplier_u32 = function.getRegs( Jit::u32 , 1 );
    function.asm_mov_literal( r_multiplier_u32 , (unsigned)JITType<T2>::Type_t::Size_t * WordSize<T2>::Size );
    int r_offset_u64 = function.getRegs( Jit::u64 , 1 );
    function.asm_mul( r_offset_u64 , r_s_p_tid_u32 , r_multiplier_u32 );
    int r_addr_u64 = function.getRegs( Jit::u64 , 1 );
    function.asm_add( r_addr_u64 , function.getSDATA() , r_offset_u64 );
    function.set_state_space( r_addr_u64 , Jit::SHARED );

    typename JITType<T2>::Type_t sdata_p( curry_t(function,r_addr_u64,1,0) );

    function.asm_cmp( Jit::CmpOp::gt , r_pred_max , r_s , r_zero );
    function.addCondBranchPredToLabel( r_pred_s , "LOOP_S_END" );

    sdata.elem(0) += sdata_p;
    //sdata.elem(0) = sdata.elem(0) + sdata.elem(0);

    function.insert_label("SYNC");

    function.asm_shr( r_s , r_s , r_one );

    function.asm_bar_sync(1);

    function.addBranchToLabel( "LOOP_S" );
    function.insert_label("LOOP_S_END");

    int r_pred_tid_ne_0 = function.getRegs( Jit::pred , 1 );

    function.asm_cmp( Jit::CmpOp::ne , r_pred_tid_ne_0 , function.getTID() , r_zero );
    function.addCondBranchPredToLabel( r_pred_tid_ne_0 , "EXIT" );

    odata.elem(0) = sdata.elem(0);

    function.insert_label("EXIT");

    if (Layout::primaryNode())
      function.write();
      
    QMP_barrier();

    CUresult ret;
    CUmodule cuModule;
    ret = cuModuleLoad(&cuModule, fname.c_str() );
    if (ret) QDP_error_exit("Error loading CUDA module '%s'",fname.c_str());

    ret = cuModuleGetFunction(&func, cuModule, "func");
    if (ret) { 
      std::cout << "Error getting function (error code = " << ret << ")\n";
      std::cout << CUDA_ERROR_DEINITIALIZED << " CUDA_ERROR_DEINITIALIZED\n";
      std::cout << CUDA_ERROR_NOT_INITIALIZED << " CUDA_ERROR_NOT_INITIALIZED\n";
      std::cout << CUDA_ERROR_INVALID_CONTEXT << " CUDA_ERROR_INVALID_CONTEXT\n";
      std::cout << CUDA_ERROR_INVALID_VALUE << " CUDA_ERROR_INVALID_VALUE\n";
      std::cout << CUDA_ERROR_NOT_FOUND << " CUDA_ERROR_NOT_FOUND\n";
      exit(1);
    }

    //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

    return func;
  }


}










#endif
