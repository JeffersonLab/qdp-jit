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
    llvm_start_new_function();

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_site_perm  = llvm_add_param< int* >(); // Siteperm  array
    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm::Value* r_lo     = llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm::Value* r_idata      = llvm_derefParam( p_idata );  // Input  array
    llvm::Value* r_odata      = llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<T2WT>::value );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_perm , r_idx );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );

    sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)

    llvm_bar_sync();

    llvm::Value* val_ntid = llvm_call_special_ntidx();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();
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
    r_pow->addIncoming( llvm_create_value(1) , entry_block );

    llvm_cond_branch( llvm_ge( r_pow , val_ntid ) , block_power_loop_exit , block_power_loop_inc );
    {
      llvm_set_insert_point(block_power_loop_inc);
      r_pow_phi = llvm_shl( r_pow , llvm_create_value(1) );
      r_pow->addIncoming( r_pow_phi , block_power_loop_inc );
      llvm_branch( block_power_loop_start );
    }

    llvm_set_insert_point(block_power_loop_exit);


    //
    // Shared memory reduction loop
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
    r_red_pow->addIncoming( r_pow , block_power_loop_exit );
    llvm_cond_branch( llvm_le( r_red_pow , llvm_create_value(0) ) , block_red_loop_end , block_red_loop_start_1 );

    llvm_set_insert_point(block_red_loop_start_1);

    llvm_cond_branch( llvm_ge( r_tidx , r_red_pow ) , block_red_loop_sync , block_red_loop_start_2 );

    llvm_set_insert_point(block_red_loop_start_2);

    llvm::Value * v = llvm_add( r_red_pow , r_tidx );
    llvm_cond_branch( llvm_ge( v , r_ntidx ) , block_red_loop_sync , block_red_loop_add );

    llvm_set_insert_point(block_red_loop_add);


    IndexDomainVector args_new;
    args_new.push_back( make_pair( Layout::sitesOnNode() , 
				   llvm_add( r_tidx , r_red_pow ) ) );  // sitesOnNode irrelevant since Scalar access later

    typename JITType<T2>::Type_t sdata_jit_plus;
    sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;    // 
    sdata_reg_plus.setup( sdata_jit_plus );

    sdata_jit += sdata_reg_plus;


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
    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_sum_ind.ptx");
  }




  template<class T1>
  CUfunction 
  function_sum_build()
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

    llvm::Value* r_idata      = llvm_derefParam( p_idata );  // Input  array
    llvm::Value* r_odata      = llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<WT>::value );


    typedef typename JITType<T1>::Type_t T1JIT;

    llvm::Value* r_idx = llvm_thread_idx();   

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_idx ) ); 

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
      zero_rep( sdata_jit );
      llvm_branch( block_zero_exit );
    }
    {
      llvm_set_insert_point(block_not_zero);
      sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)
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

    //
    // Shared memory reduction loop
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
    r_red_pow->addIncoming( r_pow , block_power_loop_exit );
    llvm_cond_branch( llvm_le( r_red_pow , llvm_create_value(0) ) , block_red_loop_end , block_red_loop_start_1 );

    llvm_set_insert_point(block_red_loop_start_1);

    llvm_cond_branch( llvm_ge( r_tidx , r_red_pow ) , block_red_loop_sync , block_red_loop_start_2 );

    llvm_set_insert_point(block_red_loop_start_2);

    llvm::Value * v = llvm_add( r_red_pow , r_tidx );
    llvm_cond_branch( llvm_ge( v , r_ntidx ) , block_red_loop_sync , block_red_loop_add );

    llvm_set_insert_point(block_red_loop_add);


    IndexDomainVector args_new;
    args_new.push_back( make_pair( Layout::sitesOnNode() , 
				   llvm_add( r_tidx , r_red_pow ) ) );  // sitesOnNode irrelevant since Scalar access later

    typename JITType<T1>::Type_t sdata_jit_plus;
    sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg_plus;    // 
    sdata_reg_plus.setup( sdata_jit_plus );

    sdata_jit += sdata_reg_plus;


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
    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_sum.ptx");
  }

} // namespace

#endif
