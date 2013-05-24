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

    llvm::outs() << "0\n";

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;
    //llvm::Type* T1type = llvm_type<WT>::value;

    ParamRef p_site_perm  = llvm_add_param< int* >(); // Siteperm  array
    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm::Value* r_lo     = llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm::Value* r_idata      = llvm_derefParam( p_idata );  // Input  array
    llvm::Value* r_odata      = llvm_derefParam( p_odata );  // output array

    llvm::outs() << "1\n";

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<T2WT>::value );

    llvm::outs() << "\nr_shared->dump() = ";
    r_shared->dump();
    llvm::outs() << "\nr_shared->getType()->dump() = ";
    r_shared->getType()->dump();
    llvm::outs() << "\n";

    //llvm::Value* r_shared = llvm_alloca( llvm_type<WT>::value , 128 );



    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_perm , r_idx );

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    llvm::outs() << "2\n";
  
    // OLatticeJIT<typename JITType<T1>::Type_t> idata(  r_idata , r_idx );   // want coal   access later
    // OLatticeJIT<typename JITType<T1>::Type_t> odata(  r_odata , r_block_idx );  // want scalar access later
    // OLatticeJIT<typename JITType<T1>::Type_t> sdata(  r_shared , r_tidx );      // want scalar access later

    // zero_rep() branch should be redundant


    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( input_layout , r_idx ) );

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later

    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );

    llvm::outs() << "3\n";

    llvm::BasicBlock * block_zero = llvm_new_basic_block();
    llvm::BasicBlock * block_not_zero = llvm_new_basic_block();
    llvm::BasicBlock * block_zero_exit = llvm_new_basic_block();
    llvm_cond_branch( llvm_ge( r_idx , r_hi ) , block_zero , block_not_zero );
    {
      llvm::outs() << "3a\n";
      llvm_set_insert_point(block_zero);
      llvm::outs() << "3b\n";
      zero_rep( sdata_jit );
      llvm::outs() << "3c\n";
      llvm_branch( block_zero_exit );
      llvm::outs() << "3d\n";
    }
    {
      llvm_set_insert_point(block_not_zero);
      llvm::outs() << "3e\n";
      sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)
      llvm::outs() << "3f\n";
      llvm_branch( block_zero_exit );
    }
    llvm_set_insert_point(block_zero_exit);

    llvm::outs() << "4\n";

    llvm_bar_sync();

    llvm::Value* val_ntid = llvm_call_special_ntidx();

    //
    // Find next power of 2 loop
    //
    llvm::BasicBlock * block_power_loop_start = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_inc = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_not_inc = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_inc_exit = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_exit = llvm_new_basic_block();
    llvm::Value* r_pow_phi;

    llvm_branch( block_power_loop_start );

    llvm_set_insert_point( block_power_loop_start );

    llvm::PHINode * r_pow = llvm_phi( llvm_type<int>::value , 2 );
    llvm::outs() << "4a\n";
    r_pow->addIncoming( llvm_create_value(1) , block_zero_exit );
    llvm::outs() << "4b\n";
    llvm::outs() << "5\n";

    llvm_cond_branch( llvm_ge( r_pow , val_ntid ) , block_power_loop_not_inc , block_power_loop_inc );
    {
      llvm_set_insert_point(block_power_loop_inc);
      r_pow_phi = llvm_shl( r_pow , llvm_create_value(1) );
      r_pow->addIncoming( r_pow_phi , block_power_loop_inc_exit );
      llvm_branch( block_power_loop_inc_exit );
    }
    {
      llvm_set_insert_point(block_power_loop_not_inc);
      llvm_branch( block_power_loop_exit );
    }
    llvm_set_insert_point(block_power_loop_inc_exit);
    llvm_branch( block_power_loop_start );

    llvm::outs() << "6\n";

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

  std::cout << __PRETTY_FUNCTION__ << ": entering\n";
  QDP_error_exit("ni");
#if 0
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    CUfunction func;

    llvm_start_new_function();

    llvm::Value* r_lo     = llvm_add_param( jit_ptx_type::s32 );
    llvm::Value* r_hi     = llvm_add_param( jit_ptx_type::s32 );

    llvm::Value* r_idx = llvm_thread_idx();

    // I'll do always a site perm here
    // Can eventually be optimized if orderedSubset
    llvm::Value* r_perm_array_addr      = llvm_add_param( jit_ptx_type::u64 );  // Site permutation array
    llvm::Value* r_idx_mul_4            = llvm_mul( r_idx , llvm_create_value(4) );
    llvm::Value* r_perm_array_addr_load = llvm_add( r_perm_array_addr , r_idx_mul_4 );
    llvm::Value* r_idx_perm             = llvm_load ( r_perm_array_addr_load , 0 , jit_ptx_type::s32 );

    llvm::Value* r_idata      = llvm_add_param( jit_ptx_type::u64 );  // Input  array
    llvm::Value* r_odata      = llvm_add_param( jit_ptx_type::u64 );  // output array
    llvm::Value* r_block_idx  = jit_geom_get_ctaidx();
    llvm::Value* r_tidx       = jit_geom_get_tidx();
    llvm::Value* r_shared     = jit_get_shared_mem_ptr();
  
    OLatticeJIT<typename JITType<T1>::Type_t> idata( r_idata , r_idx_perm );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata( r_odata , r_block_idx );  // want scalar access later
    OLatticeJIT<typename JITType<T2>::Type_t> sdata( r_shared , r_tidx );      // want scalar access later

    // zero_rep() branch should be redundant


    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   // this is stupid
    reg_idata_elem.setup( idata.elem( input_layout ) );

    jit_label_t label_zero_rep;
    jit_label_t label_zero_rep_exit;
    llvm_branch(  label_zero_rep , llvm_ge( r_idx , r_hi ) );
    sdata.elem( JitDeviceLayout::Scalar ) = reg_idata_elem; // This should do the precision conversion (SP->DP)
    llvm_branch( label_zero_rep_exit );
    llvm_label( label_zero_rep );
    zero_rep( sdata.elem( JitDeviceLayout::Scalar ) );
    llvm_label( label_zero_rep_exit );

    llvm_bar_sync( 0 );

    llvm::Value* val_ntid = jit_geom_get_ntidx();

    //
    // Find next power of 2 loop
    //
    llvm::Value* r_pred_pow(1);
    jit_label_t label_power_end;
    jit_label_t label_power_start;
    llvm_label(  label_power_start );

    llvm::Value* pred_ge = llvm_ge( r_pred_pow , val_ntid );
    llvm_branch(  label_power_end , pred_ge );
    llvm::Value* new_pred = llvm_shl( r_pred_pow , llvm_create_value(1) );
    llvm_mov( r_pred_pow , new_pred );
  
    llvm_branch(  label_power_start );
    llvm_label(  label_power_end );

    new_pred = llvm_shr( r_pred_pow , llvm_create_value(1) );
    llvm_mov( r_pred_pow , new_pred );

    //
    // Shared memory reduction loop
    //
    jit_label_t label_loop_start;
    jit_label_t label_loop_sync;
    jit_label_t label_loop_end;
    llvm_label(  label_loop_start );

    llvm::Value* pred_branch_end = llvm_le( r_pred_pow , llvm_create_value(0) );
    llvm_branch(  label_loop_end , pred_branch_end );

    llvm::Value* pred_branch_sync = llvm_ge( jit_geom_get_tidx() , r_pred_pow );
    llvm_branch(  label_loop_sync , pred_branch_sync );

    llvm::Value* val_s_plus_tid = llvm_add( r_pred_pow , jit_geom_get_tidx() );
    llvm::Value* pred_branch_sync2 = llvm_ge( val_s_plus_tid , jit_geom_get_ntidx() );
    llvm_branch(  label_loop_sync , pred_branch_sync2 );

    OLatticeJIT<typename JITType<T2>::Type_t> sdata_plus_s(  r_shared , 
							    llvm_add( r_tidx , r_pred_pow ) );

    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_plus_s_elem;   // this is stupid
    sdata_plus_s_elem.setup( sdata_plus_s.elem( JitDeviceLayout::Scalar ) );
    sdata.elem( JitDeviceLayout::Scalar ) += sdata_plus_s_elem;

    llvm_label(  label_loop_sync );  
    llvm_bar_sync(  0 );

    new_pred = llvm_shr( r_pred_pow , llvm_create_value(1) );
    llvm_mov( r_pred_pow , new_pred );

    llvm_branch(  label_loop_start );
  
    llvm_label(  label_loop_end );  

    jit_label_t label_exit;
    llvm::Value* pred_branch_exit = llvm_ne( jit_geom_get_tidx() , llvm_create_value(0) );
    llvm_branch(  label_exit , pred_branch_exit );

    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata.elem( JitDeviceLayout::Scalar ) );
    odata.elem( JitDeviceLayout::Scalar ) = sdata_reg;

    llvm_label(  label_exit );

    llvm_exit();

    return llvm_get_cufunction("ptx_sum_ind.ptx");
#endif
  }




  template<class T1>
  CUfunction 
  function_sum_build()
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";
    llvm_start_new_function();

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t WT;
    //llvm::Type* T1type = llvm_type<WT>::value;

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

  
    // OLatticeJIT<typename JITType<T1>::Type_t> idata(  r_idata , r_idx );   // want coal   access later
    // OLatticeJIT<typename JITType<T1>::Type_t> odata(  r_odata , r_block_idx );  // want scalar access later
    // OLatticeJIT<typename JITType<T1>::Type_t> sdata(  r_shared , r_tidx );      // want scalar access later

    // zero_rep() branch should be redundant


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
    llvm::BasicBlock * block_power_loop_not_inc = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_inc_exit = llvm_new_basic_block();
    llvm::BasicBlock * block_power_loop_exit = llvm_new_basic_block();
    llvm::Value* r_pow_phi;

    llvm_branch( block_power_loop_start );

    llvm_set_insert_point( block_power_loop_start );

    llvm::PHINode * r_pow = llvm_phi( llvm_type<int>::value , 2 );
    r_pow->addIncoming( llvm_create_value(1) , block_zero_exit );
    r_pow->addIncoming( r_pow_phi , block_power_loop_inc_exit );

    llvm_cond_branch( llvm_ge( r_pow , val_ntid ) , block_power_loop_not_inc , block_power_loop_inc );
    {
      llvm_set_insert_point(block_power_loop_inc);
      r_pow_phi = llvm_shl( r_pow , llvm_create_value(1) );
      llvm_branch( block_power_loop_inc_exit );
    }
    {
      llvm_set_insert_point(block_power_loop_not_inc);
      llvm_branch( block_power_loop_inc_exit );
    }
    llvm_set_insert_point(block_power_loop_inc_exit);
    llvm_branch( block_power_loop_start );

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

#if 0
    //
    // Shared memory reduction loop
    //
    jit_label_t label_loop_start;
    jit_label_t label_loop_sync;
    jit_label_t label_loop_end;
    llvm_label(  label_loop_start );

    llvm::Value* pred_branch_end = llvm_le( r_pred_pow , llvm_create_value(0) );
    llvm_branch(  label_loop_end , pred_branch_end );

    llvm::Value* pred_branch_sync = llvm_ge( jit_geom_get_tidx() , r_pred_pow );
    llvm_branch(  label_loop_sync , pred_branch_sync );

    llvm::Value* val_s_plus_tid = llvm_add( r_pred_pow , jit_geom_get_tidx() );
    llvm::Value* pred_branch_sync2 = llvm_ge( val_s_plus_tid , jit_geom_get_ntidx() );
    llvm_branch(  label_loop_sync , pred_branch_sync2 );

    OLatticeJIT<typename JITType<T1>::Type_t> sdata_plus_s(  r_shared , 
							    llvm_add( r_tidx , r_pred_pow ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_plus_s_elem;   // this is stupid
    sdata_plus_s_elem.setup( sdata_plus_s.elem( JitDeviceLayout::Scalar ) );
    sdata.elem( JitDeviceLayout::Scalar ) += sdata_plus_s_elem;

    llvm_label(  label_loop_sync );  
    llvm_bar_sync(  0 );

    new_pred = llvm_shr( r_pred_pow , llvm_create_value(1) );
    llvm_mov( r_pred_pow , new_pred );

    llvm_branch(  label_loop_start );
  
    llvm_label(  label_loop_end );  

    jit_label_t label_exit;
    llvm::Value* pred_branch_exit = llvm_ne( jit_geom_get_tidx() , llvm_create_value(0) );
    llvm_branch(  label_exit , pred_branch_exit );

    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   // this is stupid
    sdata_reg.setup( sdata.elem( JitDeviceLayout::Scalar ) );
    odata.elem( JitDeviceLayout::Scalar ) = sdata_reg;
#endif
  }



}

#endif
