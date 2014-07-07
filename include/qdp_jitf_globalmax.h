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
    llvm_start_new_function();

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t WT;

    ParamRef p_idata      = llvm_add_param< WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata(  p_odata );   // want scalar access later

    llvm::Value* r_lo     = llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm::Value* r_idx = llvm_thread_idx();   
    // llvm_cond_exit( llvm_ge( r_idx , r_hi ) );  // We can't exit, because we have to initialize shared data

    llvm::Value* r_idata      = llvm_derefParam( p_idata );  // Input  array
    llvm::Value* r_odata      = llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<WT>::value );


    typedef typename JITType<T1>::Type_t T1JIT;


    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_idx ) ); // GlobalMax only on scalar types

    // We use the 1st element of the input array
    // to fill out the shared memory at the index
    // positions that reach out
    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_1st_elem;
    reg_idata_1st_elem.setup( idata.elem( JitDeviceLayout::Scalar , llvm_create_value(0) ) ); 

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later

    T1JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );

    llvm::BasicBlock * block_zero = llvm_new_basic_block();
    llvm::BasicBlock * block_not_zero = llvm_new_basic_block();
    llvm::BasicBlock * block_zero_exit = llvm_new_basic_block();
    llvm_cond_branch( llvm_ge( r_idx , r_hi ) , block_zero , block_not_zero );
    {
      llvm_set_insert_point(block_zero);
      sdata_jit = reg_idata_1st_elem;
      llvm_branch( block_zero_exit );
    }
    {
      llvm_set_insert_point(block_not_zero);
      sdata_jit = reg_idata_elem;
      llvm_branch( block_zero_exit );
    }
    llvm_set_insert_point(block_zero_exit);

    llvm_bar_sync();

    llvm::Value* val_ntid = llvm_call_special_ntidx();

    //
    // Find next power of 2 loop
    //
    llvm::BasicBlock * block_power_loop_start = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_inc = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_exit = llvm_new_basic_block();
    llvm::Value* r_pow_phi;

    llvm_branch( block_power_loop_start );

    llvm_set_insert_point( block_power_loop_start );

    llvm::PHINode * r_pow = llvm_phi( llvm_type<int>::value , 2 );
    r_pow->addIncoming( llvm_create_value(1) , block_zero_exit );

    llvm_cond_branch( llvm_ge( r_pow , val_ntid ) , block_power_loop_exit , block_power_loop_inc );
    {
      llvm_set_insert_point(block_power_loop_inc);
      r_pow_phi = llvm_shl( r_pow , llvm_create_value(1) );
      r_pow->addIncoming( r_pow_phi , block_power_loop_inc );
      llvm_branch( block_power_loop_start );
    }

    llvm_set_insert_point(block_power_loop_exit);

    llvm::Value* r_pow_shr1 = llvm_shr( r_pow , llvm_create_value(1) );

    //
    // Shared memory maximizing loop
    //
    llvm::BasicBlock * block_red_loop_start = llvm_new_basic_block();
    llvm::BasicBlock * block_red_loop_start_1 = llvm_new_basic_block();
    llvm::BasicBlock * block_red_loop_start_2 = llvm_new_basic_block();
    llvm::BasicBlock * block_red_loop_add = llvm_new_basic_block();
    llvm::BasicBlock * block_red_loop_sync = llvm_new_basic_block();
    llvm::BasicBlock * block_red_loop_end = llvm_new_basic_block();

    llvm_branch( block_red_loop_start );
    llvm_set_insert_point(block_red_loop_start);
    
    llvm::PHINode * r_red_pow = llvm_phi( llvm_type<int>::value , 2 );    
    r_red_pow->addIncoming( r_pow_shr1 , block_power_loop_exit );
    llvm_cond_branch( llvm_le( r_red_pow , llvm_create_value(0) ) , block_red_loop_end , block_red_loop_start_1 );

    llvm_set_insert_point(block_red_loop_start_1);

    llvm_cond_branch( llvm_ge( r_tidx , r_red_pow ) , block_red_loop_sync , block_red_loop_start_2 );

    llvm_set_insert_point(block_red_loop_start_2);

    llvm::Value * v = llvm_add( r_red_pow , r_tidx ); // target index to compare index r_tidx with
    llvm_cond_branch( llvm_ge( v , r_ntidx ) , block_red_loop_sync , block_red_loop_add );

    llvm_set_insert_point(block_red_loop_add);


    IndexDomainVector args_new;
    args_new.push_back( make_pair( Layout::sitesOnNode(),v ) );  // sitesOnNode irrelevant since Scalar access later

    typename JITType<T1>::Type_t sdata_jit_plus;
    sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg_plus;    
    sdata_reg_plus.setup( sdata_jit_plus );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   
    sdata_reg.setup( sdata_jit );

    sdata_jit = where( sdata_reg > sdata_reg_plus , sdata_reg , sdata_reg_plus );

    //sdata_jit += sdata_reg_plus;


    llvm_branch( block_red_loop_sync );

    llvm_set_insert_point(block_red_loop_sync);
    llvm_bar_sync();
    llvm::Value* pow_1 = llvm_shr( r_red_pow , llvm_create_value(1) );
    r_red_pow->addIncoming( pow_1 , block_red_loop_sync );

    llvm_branch( block_red_loop_start );

    llvm_set_insert_point(block_red_loop_end);

    llvm::BasicBlock * block_store_global = llvm_new_basic_block();
    llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();
    llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
		      block_store_global , 
		      block_not_store_global );
    llvm_set_insert_point(block_store_global);
    //typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_max.ptx");
  }




#if 0
  QDP_error_exit("ni");
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
  }
#endif


}
#endif
