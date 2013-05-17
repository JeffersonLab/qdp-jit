#ifndef QDP_JITF_GLOBMAX_H
#define QDP_JITF_GLOBMAX_H

// this code is not used


#include "qmp.h"

namespace QDP {


  void function_global_max_exec( CUfunction function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 void *d_idata, void *d_odata);


  template<class T1>
  CUfunction 
  function_global_max_build()
  {
  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    llvm_start_new_function();

    llvm::Value* r_lo     = llvm_add_param( jit_ptx_type::s32 );
    llvm::Value* r_hi     = llvm_add_param( jit_ptx_type::s32 );

    llvm::Value* r_idx = llvm_thread_idx();  

    llvm::Value* r_out_of_range       = llvm_ge( r_idx , r_hi );
    llvm_exit( r_out_of_range );

    llvm::Value* r_idata      = llvm_add_param( jit_ptx_type::u64 );  // Input  array
    llvm::Value* r_odata      = llvm_add_param( jit_ptx_type::u64 );  // output array
    llvm::Value* r_block_idx  = jit_geom_get_ctaidx();
    llvm::Value* r_tidx       = jit_geom_get_tidx();
    llvm::Value* r_shared     = jit_get_shared_mem_ptr();
  
    OLatticeJIT<typename JITType<T1>::Type_t> idata( r_idata , r_idx );       // want coal/scalar (templ. param)
    OLatticeJIT<typename JITType<T1>::Type_t> odata( r_odata , r_block_idx ); // want scalar access later
    OLatticeJIT<typename JITType<T1>::Type_t> sdata( r_shared , r_tidx );     // want scalar access later

    // zero_rep() branch should be redundant


    typename REGType< typename JITType<T1>::Type_t >::Type_t idata_reg; // this is stupid
    idata_reg.setup( idata.elem( JitDeviceLayout::Scalar ) );            // Scalar is fine, because it's a scalar data type
    sdata.elem( JitDeviceLayout::Scalar ) = idata_reg;

    llvm_bar_sync( 0 );

    llvm::Value* val_ntid = jit_geom_get_ntidx();

    //
    // Find next power of 2 loop
    //
    llvm::Value* r_pred_pow(1);
    jit_label_t label_power_end;
    jit_label_t label_power_start;
    llvm_label( label_power_start );

    llvm::Value* pred_ge = llvm_ge( r_pred_pow , val_ntid );
    llvm_branch( label_power_end , pred_ge );
    llvm::Value* new_pred = llvm_shl( r_pred_pow , llvm_create_value(1) );
    llvm_mov( r_pred_pow , new_pred );
  
    llvm_branch( label_power_start );
    llvm_label( label_power_end );

    new_pred = llvm_shr( r_pred_pow , llvm_create_value(1) );
    llvm_mov( r_pred_pow , new_pred );

    //
    // Shared memory maximizing loop
    //
    jit_label_t label_loop_start;
    jit_label_t label_loop_sync;
    jit_label_t label_loop_end;
    llvm_label( label_loop_start );

    llvm::Value* pred_branch_end = llvm_le( r_pred_pow , llvm_create_value(0) );
    llvm_branch( label_loop_end , pred_branch_end );

    llvm::Value* pred_branch_sync = llvm_ge( jit_geom_get_tidx() , r_pred_pow );
    llvm_branch( label_loop_sync , pred_branch_sync );

    llvm::Value* val_s_plus_tid = llvm_add( r_pred_pow , jit_geom_get_tidx() );
    llvm::Value* pred_branch_sync2 = llvm_ge( val_s_plus_tid , jit_geom_get_ntidx() );
    llvm_branch( label_loop_sync , pred_branch_sync2 );

    OLatticeJIT<typename JITType<T1>::Type_t> sdata_plus_s( r_shared , 
							    llvm_add( r_tidx , r_pred_pow ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_plus_s_elem;   // this is stupid
    sdata_plus_s_elem.setup( sdata_plus_s.elem( JitDeviceLayout::Scalar ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata.elem( JitDeviceLayout::Scalar ) );

    sdata.elem( JitDeviceLayout::Scalar ) = where( sdata_reg > sdata_plus_s_elem , sdata_reg , sdata_plus_s_elem );

    llvm_label( label_loop_sync );
    llvm_bar_sync( 0 );

    new_pred = llvm_shr( r_pred_pow , llvm_create_value(1) );
    llvm_mov( r_pred_pow , new_pred );

    llvm_branch( label_loop_start );
  
    llvm_label( label_loop_end );  

    jit_label_t label_exit;
    llvm::Value* pred_branch_exit = llvm_ne( jit_geom_get_tidx() , llvm_create_value(0) );
    llvm_branch( label_exit , pred_branch_exit );

    sdata_reg.setup( sdata.elem( JitDeviceLayout::Scalar ) );
    odata.elem( JitDeviceLayout::Scalar ) = sdata_reg;

    llvm_label( label_exit );

    llvm_exit();

    return llvm_get_cufunction("ptx_global_max.ptx");
#endif
  }


}
#endif
