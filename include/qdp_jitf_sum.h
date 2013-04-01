#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

#include "qmp.h"

namespace QDP {

void function_sum_ind_coal_exec( CUfunction function, 
				 int size, int threads, int blocks, int shared_mem_usage,
				 void *d_idata, void *d_odata, void *siteTable);

void function_sum_exec( CUfunction function, 
			int size, int threads, int blocks, int shared_mem_usage,
			void *d_idata, void *d_odata);

  // T1 input
  // T2 output
  template<class T1,class T2>
  CUfunction 
  function_sum_ind_coal_build()
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    const char * fname = "ptx_sum_ind.ptx";
    jit_function_t function = jit_create_function( fname );

    jit_value_t r_lo     = jit_add_param( function , jit_ptx_type::s32 );
    jit_value_t r_hi     = jit_add_param( function , jit_ptx_type::s32 );

    jit_value_t r_idx = jit_geom_get_linear_th_idx( function );  

    jit_value_t r_out_of_range       = jit_ins_ge( r_idx , r_hi );
    jit_ins_exit( function , r_out_of_range );

    // I'll do always a site perm here
    // Can eventually be optimized if orderedSubset
    jit_value_t r_perm_array_addr      = jit_add_param( function , jit_ptx_type::u64 );  // Site permutation array
    jit_value_t r_idx_mul_4            = jit_ins_mul( r_idx , jit_val_create_const_int(4) );
    jit_value_t r_perm_array_addr_load = jit_ins_add( r_perm_array_addr , r_idx_mul_4 );
    jit_value_t r_idx_perm             = jit_ins_load ( r_perm_array_addr_load , 0 , jit_ptx_type::s32 );

    jit_value_t r_idata      = jit_add_param( function , jit_ptx_type::u64 );  // Input  array
    jit_value_t r_odata      = jit_add_param( function , jit_ptx_type::u64 );  // output array
    jit_value_t r_block_idx  = jit_geom_get_ctaidx( function );
    jit_value_t r_tidx       = jit_geom_get_tidx( function );
    jit_value_t r_shared     = jit_get_shared_mem_ptr( function );
  
    OLatticeJIT<typename JITType<T1>::Type_t> idata( function , r_idata , r_idx_perm );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata( function , r_odata , r_block_idx );  // want scalar access later
    OLatticeJIT<typename JITType<T2>::Type_t> sdata( function , r_shared , r_tidx );      // want scalar access later

    // zero_rep() branch should be redundant


    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( QDPTypeJITBase::Coalesced ) );
    sdata.elem( QDPTypeJITBase::Scalar ) = reg_idata_elem; // This should do the precision conversion (SP->DP)

    jit_ins_bar_sync( function , 0 );

    jit_value_t val_ntid = jit_geom_get_ntidx(function);

    //
    // Find next power of 2 loop
    //
    jit_value_t r_pred_pow = jit_val_create_from_const( function , jit_ptx_type::u32 , 1 );
    jit_label_t label_power_end;
    jit_label_t label_power_start;
    jit_ins_label( function , label_power_start );

    jit_value_t pred_ge = jit_ins_ge( r_pred_pow , val_ntid );
    jit_ins_branch( function , label_power_end , pred_ge );
    jit_value_t new_pred = jit_ins_shl( r_pred_pow , jit_val_create_const_int(1) );
    jit_ins_mov_no_create( r_pred_pow , new_pred );
  
    jit_ins_branch( function , label_power_start );
    jit_ins_label( function , label_power_end );

    new_pred = jit_ins_shr( r_pred_pow , jit_val_create_const_int(1) );
    jit_ins_mov_no_create( r_pred_pow , new_pred );

    //
    // Shared memory reduction loop
    //
    jit_label_t label_loop_start;
    jit_label_t label_loop_sync;
    jit_label_t label_loop_end;
    jit_ins_label( function , label_loop_start );

    jit_value_t pred_branch_end = jit_ins_le( r_pred_pow , jit_val_create_const_int(0) );
    jit_ins_branch( function , label_loop_end , pred_branch_end );

    jit_value_t pred_branch_sync = jit_ins_ge( jit_geom_get_tidx(function) , r_pred_pow );
    jit_ins_branch( function , label_loop_sync , pred_branch_sync );

    jit_value_t val_s_plus_tid = jit_ins_add( r_pred_pow , jit_geom_get_tidx(function) );
    jit_value_t pred_branch_sync2 = jit_ins_ge( val_s_plus_tid , jit_geom_get_ntidx(function) );
    jit_ins_branch( function , label_loop_sync , pred_branch_sync2 );

    OLatticeJIT<typename JITType<T2>::Type_t> sdata_plus_s( function , r_shared , 
							    jit_ins_add( r_tidx , r_pred_pow ) );

    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_plus_s_elem;   // this is stupid
    sdata_plus_s_elem.setup( sdata_plus_s.elem( QDPTypeJITBase::Scalar ) );
    sdata.elem( QDPTypeJITBase::Scalar ) += sdata_plus_s_elem;

    jit_ins_label( function , label_loop_sync );  
    jit_ins_bar_sync( function , 0 );

    new_pred = jit_ins_shr( r_pred_pow , jit_val_create_const_int(1) );
    jit_ins_mov_no_create( r_pred_pow , new_pred );

    jit_ins_branch( function , label_loop_start );
  
    jit_ins_label( function , label_loop_end );  

    jit_label_t label_exit;
    jit_value_t pred_branch_exit = jit_ins_ne( jit_geom_get_tidx(function) , jit_val_create_const_int(0) );
    jit_ins_branch( function , label_exit , pred_branch_exit );

    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata.elem( QDPTypeJITBase::Scalar ) );
    odata.elem( QDPTypeJITBase::Scalar ) = sdata_reg;

    jit_ins_label( function , label_exit );

#if 1
    if (Layout::primaryNode())
      function->write();
#endif
    //  assert(!"ni");

    QMP_barrier();

    CUresult ret;
    CUmodule cuModule;
    ret = cuModuleLoad( &cuModule , fname );
    if (ret) QDP_error_exit( "Error loading CUDA module '%s'" , fname );

    ret = cuModuleGetFunction(&func, cuModule, "function");
    if (ret) { std::cout << "Error getting function\n"; exit(1); }

    //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

    return func;
  }




  template<class T1>
  CUfunction 
  function_sum_build()
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    const char * fname = "ptx_sum.ptx";
    jit_function_t function = jit_create_function( fname );

    jit_value_t r_lo     = jit_add_param( function , jit_ptx_type::s32 );
    jit_value_t r_hi     = jit_add_param( function , jit_ptx_type::s32 );

    jit_value_t r_idx = jit_geom_get_linear_th_idx( function );  

    jit_value_t r_out_of_range       = jit_ins_ge( r_idx , r_hi );
    jit_ins_exit( function , r_out_of_range );

    jit_value_t r_idata      = jit_add_param( function , jit_ptx_type::u64 );  // Input  array
    jit_value_t r_odata      = jit_add_param( function , jit_ptx_type::u64 );  // output array
    jit_value_t r_block_idx  = jit_geom_get_ctaidx( function );
    jit_value_t r_tidx       = jit_geom_get_tidx( function );
    jit_value_t r_shared     = jit_get_shared_mem_ptr( function );
  
    OLatticeJIT<typename JITType<T1>::Type_t> idata( function , r_idata , r_idx );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata( function , r_odata , r_block_idx );  // want scalar access later
    OLatticeJIT<typename JITType<T1>::Type_t> sdata( function , r_shared , r_tidx );      // want scalar access later

    // zero_rep() branch should be redundant


    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( QDPTypeJITBase::Scalar ) );
    sdata.elem( QDPTypeJITBase::Scalar ) = reg_idata_elem;

    jit_ins_bar_sync( function , 0 );

    jit_value_t val_ntid = jit_geom_get_ntidx(function);

    //
    // Find next power of 2 loop
    //
    jit_value_t r_pred_pow = jit_val_create_from_const( function , jit_ptx_type::u32 , 1 );
    jit_label_t label_power_end;
    jit_label_t label_power_start;
    jit_ins_label( function , label_power_start );

    jit_value_t pred_ge = jit_ins_ge( r_pred_pow , val_ntid );
    jit_ins_branch( function , label_power_end , pred_ge );
    jit_value_t new_pred = jit_ins_shl( r_pred_pow , jit_val_create_const_int(1) );
    jit_ins_mov_no_create( r_pred_pow , new_pred );
  
    jit_ins_branch( function , label_power_start );
    jit_ins_label( function , label_power_end );

    new_pred = jit_ins_shr( r_pred_pow , jit_val_create_const_int(1) );
    jit_ins_mov_no_create( r_pred_pow , new_pred );

    //
    // Shared memory reduction loop
    //
    jit_label_t label_loop_start;
    jit_label_t label_loop_sync;
    jit_label_t label_loop_end;
    jit_ins_label( function , label_loop_start );

    jit_value_t pred_branch_end = jit_ins_le( r_pred_pow , jit_val_create_const_int(0) );
    jit_ins_branch( function , label_loop_end , pred_branch_end );

    jit_value_t pred_branch_sync = jit_ins_ge( jit_geom_get_tidx(function) , r_pred_pow );
    jit_ins_branch( function , label_loop_sync , pred_branch_sync );

    jit_value_t val_s_plus_tid = jit_ins_add( r_pred_pow , jit_geom_get_tidx(function) );
    jit_value_t pred_branch_sync2 = jit_ins_ge( val_s_plus_tid , jit_geom_get_ntidx(function) );
    jit_ins_branch( function , label_loop_sync , pred_branch_sync2 );

    OLatticeJIT<typename JITType<T1>::Type_t> sdata_plus_s( function , r_shared , 
							    jit_ins_add( r_tidx , r_pred_pow ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_plus_s_elem;   // this is stupid
    sdata_plus_s_elem.setup( sdata_plus_s.elem( QDPTypeJITBase::Scalar ) );
    sdata.elem( QDPTypeJITBase::Scalar ) += sdata_plus_s_elem;

    jit_ins_label( function , label_loop_sync );  
    jit_ins_bar_sync( function , 0 );

    new_pred = jit_ins_shr( r_pred_pow , jit_val_create_const_int(1) );
    jit_ins_mov_no_create( r_pred_pow , new_pred );

    jit_ins_branch( function , label_loop_start );
  
    jit_ins_label( function , label_loop_end );  

    jit_label_t label_exit;
    jit_value_t pred_branch_exit = jit_ins_ne( jit_geom_get_tidx(function) , jit_val_create_const_int(0) );
    jit_ins_branch( function , label_exit , pred_branch_exit );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata.elem( QDPTypeJITBase::Scalar ) );
    odata.elem( QDPTypeJITBase::Scalar ) = sdata_reg;

    jit_ins_label( function , label_exit );

#if 1
    if (Layout::primaryNode())
      function->write();
#endif
    //  assert(!"ni");

    QMP_barrier();

    CUresult ret;
    CUmodule cuModule;
    ret = cuModuleLoad( &cuModule , fname );
    if (ret) QDP_error_exit( "Error loading CUDA module '%s'" , fname );

    ret = cuModuleGetFunction(&func, cuModule, "function");
    if (ret) { std::cout << "Error getting function\n"; exit(1); }

    //std::cout << __PRETTY_FUNCTION__ << ": exiting\n";

    return func;
  }



}

#endif