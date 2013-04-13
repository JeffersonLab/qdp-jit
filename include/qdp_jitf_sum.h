#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

#include "qmp.h"

namespace QDP {

void function_sum_ind_exec( CUfunction function, 
			    int size, int threads, int blocks, int shared_mem_usage,
			    void *d_idata, void *d_odata, void *siteTable);

void function_sum_exec( CUfunction function, 
			int size, int threads, int blocks, int shared_mem_usage,
			void *d_idata, void *d_odata);

  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout >
  CUfunction 
  function_sum_ind_build()
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    jit_start_new_function();

    jit_value r_lo     = jit_add_param( jit_ptx_type::s32 );
    jit_value r_hi     = jit_add_param( jit_ptx_type::s32 );

    jit_value r_idx = jit_geom_get_linear_th_idx();

    // I'll do always a site perm here
    // Can eventually be optimized if orderedSubset
    jit_value r_perm_array_addr      = jit_add_param( jit_ptx_type::u64 );  // Site permutation array
    jit_value r_idx_mul_4            = jit_ins_mul( r_idx , jit_value(4) );
    jit_value r_perm_array_addr_load = jit_ins_add( r_perm_array_addr , r_idx_mul_4 );
    jit_value r_idx_perm             = jit_ins_load ( r_perm_array_addr_load , 0 , jit_ptx_type::s32 );

    jit_value r_idata      = jit_add_param( jit_ptx_type::u64 );  // Input  array
    jit_value r_odata      = jit_add_param( jit_ptx_type::u64 );  // output array
    jit_value r_block_idx  = jit_geom_get_ctaidx();
    jit_value r_tidx       = jit_geom_get_tidx();
    jit_value r_shared     = jit_get_shared_mem_ptr();
  
    OLatticeJIT<typename JITType<T1>::Type_t> idata( r_idata , r_idx_perm );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata( r_odata , r_block_idx );  // want scalar access later
    OLatticeJIT<typename JITType<T2>::Type_t> sdata( r_shared , r_tidx );      // want scalar access later

    // zero_rep() branch should be redundant


    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( input_layout ) );

    jit_label_t label_zero_rep;
    jit_label_t label_zero_rep_exit;
    jit_ins_branch(  label_zero_rep , jit_ins_ge( r_idx , r_hi ) );
    sdata.elem( JitDeviceLayout::Scalar ) = reg_idata_elem; // This should do the precision conversion (SP->DP)
    jit_ins_branch( label_zero_rep_exit );
    jit_ins_label( label_zero_rep );
    zero_rep( sdata.elem( JitDeviceLayout::Scalar ) );
    jit_ins_label( label_zero_rep_exit );

    jit_ins_bar_sync( 0 );

    jit_value val_ntid = jit_geom_get_ntidx();

    //
    // Find next power of 2 loop
    //
    jit_value r_pred_pow(1);
    jit_label_t label_power_end;
    jit_label_t label_power_start;
    jit_ins_label(  label_power_start );

    jit_value pred_ge = jit_ins_ge( r_pred_pow , val_ntid );
    jit_ins_branch(  label_power_end , pred_ge );
    jit_value new_pred = jit_ins_shl( r_pred_pow , jit_value(1) );
    jit_ins_mov( r_pred_pow , new_pred );
  
    jit_ins_branch(  label_power_start );
    jit_ins_label(  label_power_end );

    new_pred = jit_ins_shr( r_pred_pow , jit_value(1) );
    jit_ins_mov( r_pred_pow , new_pred );

    //
    // Shared memory reduction loop
    //
    jit_label_t label_loop_start;
    jit_label_t label_loop_sync;
    jit_label_t label_loop_end;
    jit_ins_label(  label_loop_start );

    jit_value pred_branch_end = jit_ins_le( r_pred_pow , jit_value(0) );
    jit_ins_branch(  label_loop_end , pred_branch_end );

    jit_value pred_branch_sync = jit_ins_ge( jit_geom_get_tidx() , r_pred_pow );
    jit_ins_branch(  label_loop_sync , pred_branch_sync );

    jit_value val_s_plus_tid = jit_ins_add( r_pred_pow , jit_geom_get_tidx() );
    jit_value pred_branch_sync2 = jit_ins_ge( val_s_plus_tid , jit_geom_get_ntidx() );
    jit_ins_branch(  label_loop_sync , pred_branch_sync2 );

    OLatticeJIT<typename JITType<T2>::Type_t> sdata_plus_s(  r_shared , 
							    jit_ins_add( r_tidx , r_pred_pow ) );

    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_plus_s_elem;   // this is stupid
    sdata_plus_s_elem.setup( sdata_plus_s.elem( JitDeviceLayout::Scalar ) );
    sdata.elem( JitDeviceLayout::Scalar ) += sdata_plus_s_elem;

    jit_ins_label(  label_loop_sync );  
    jit_ins_bar_sync(  0 );

    new_pred = jit_ins_shr( r_pred_pow , jit_value(1) );
    jit_ins_mov( r_pred_pow , new_pred );

    jit_ins_branch(  label_loop_start );
  
    jit_ins_label(  label_loop_end );  

    jit_label_t label_exit;
    jit_value pred_branch_exit = jit_ins_ne( jit_geom_get_tidx() , jit_value(0) );
    jit_ins_branch(  label_exit , pred_branch_exit );

    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata.elem( JitDeviceLayout::Scalar ) );
    odata.elem( JitDeviceLayout::Scalar ) = sdata_reg;

    jit_ins_label(  label_exit );

    return jit_get_cufunction("ptx_sum_ind.ptx");
  }




  template<class T1>
  CUfunction 
  function_sum_build()
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    jit_start_new_function();

    jit_value r_lo     = jit_add_param(  jit_ptx_type::s32 );
    jit_value r_hi     = jit_add_param(  jit_ptx_type::s32 );

    jit_value r_idx = jit_geom_get_linear_th_idx();  


    jit_value r_idata      = jit_add_param(  jit_ptx_type::u64 );  // Input  array
    jit_value r_odata      = jit_add_param(  jit_ptx_type::u64 );  // output array
    jit_value r_block_idx  = jit_geom_get_ctaidx();
    jit_value r_tidx       = jit_geom_get_tidx();
    jit_value r_shared     = jit_get_shared_mem_ptr();
  
    OLatticeJIT<typename JITType<T1>::Type_t> idata(  r_idata , r_idx );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata(  r_odata , r_block_idx );  // want scalar access later
    OLatticeJIT<typename JITType<T1>::Type_t> sdata(  r_shared , r_tidx );      // want scalar access later

    // zero_rep() branch should be redundant


    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar ) ); 

    jit_label_t label_zero_rep;
    jit_label_t label_zero_rep_exit;
    jit_ins_branch(  label_zero_rep , jit_ins_ge( r_idx , r_hi ) );
    sdata.elem( JitDeviceLayout::Scalar ) = reg_idata_elem; // This should do the precision conversion (SP->DP)
    jit_ins_branch( label_zero_rep_exit );
    jit_ins_label( label_zero_rep );
    zero_rep( sdata.elem( JitDeviceLayout::Scalar ) );
    jit_ins_label( label_zero_rep_exit );

    /* 
    /* sdata.elem( JitDeviceLayout::Scalar ) = reg_idata_elem; */

    jit_ins_bar_sync(  0 );

    jit_value val_ntid = jit_geom_get_ntidx();

    //
    // Find next power of 2 loop
    //
    jit_value r_pred_pow(1);
    jit_label_t label_power_end;
    jit_label_t label_power_start;
    jit_ins_label(  label_power_start );

    jit_value pred_ge = jit_ins_ge( r_pred_pow , val_ntid );
    jit_ins_branch(  label_power_end , pred_ge );
    jit_value new_pred = jit_ins_shl( r_pred_pow , jit_value(1) );
    jit_ins_mov( r_pred_pow , new_pred );
  
    jit_ins_branch(  label_power_start );
    jit_ins_label(  label_power_end );

    new_pred = jit_ins_shr( r_pred_pow , jit_value(1) );
    jit_ins_mov( r_pred_pow , new_pred );

    //
    // Shared memory reduction loop
    //
    jit_label_t label_loop_start;
    jit_label_t label_loop_sync;
    jit_label_t label_loop_end;
    jit_ins_label(  label_loop_start );

    jit_value pred_branch_end = jit_ins_le( r_pred_pow , jit_value(0) );
    jit_ins_branch(  label_loop_end , pred_branch_end );

    jit_value pred_branch_sync = jit_ins_ge( jit_geom_get_tidx() , r_pred_pow );
    jit_ins_branch(  label_loop_sync , pred_branch_sync );

    jit_value val_s_plus_tid = jit_ins_add( r_pred_pow , jit_geom_get_tidx() );
    jit_value pred_branch_sync2 = jit_ins_ge( val_s_plus_tid , jit_geom_get_ntidx() );
    jit_ins_branch(  label_loop_sync , pred_branch_sync2 );

    OLatticeJIT<typename JITType<T1>::Type_t> sdata_plus_s(  r_shared , 
							    jit_ins_add( r_tidx , r_pred_pow ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_plus_s_elem;   // this is stupid
    sdata_plus_s_elem.setup( sdata_plus_s.elem( JitDeviceLayout::Scalar ) );
    sdata.elem( JitDeviceLayout::Scalar ) += sdata_plus_s_elem;

    jit_ins_label(  label_loop_sync );  
    jit_ins_bar_sync(  0 );

    new_pred = jit_ins_shr( r_pred_pow , jit_value(1) );
    jit_ins_mov( r_pred_pow , new_pred );

    jit_ins_branch(  label_loop_start );
  
    jit_ins_label(  label_loop_end );  

    jit_label_t label_exit;
    jit_value pred_branch_exit = jit_ins_ne( jit_geom_get_tidx() , jit_value(0) );
    jit_ins_branch(  label_exit , pred_branch_exit );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata.elem( JitDeviceLayout::Scalar ) );
    odata.elem( JitDeviceLayout::Scalar ) = sdata_reg;

    jit_ins_label(  label_exit );

    return jit_get_cufunction("ptx_sum.ptx");
  }



}

#endif
