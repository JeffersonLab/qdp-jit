#ifndef QDP_JITF_SUM_H
#define QDP_JITF_SUM_H

//#include "qmp.h"

namespace QDP {

  void function_sum_convert_ind_exec( JitFunction function, 
				      int size, int threads, int blocks, int shared_mem_usage,
				      int in_id, int out_id, int siteTableId );

  void function_sum_convert_exec( JitFunction function, 
				  int size, int threads, int blocks, int shared_mem_usage,
				  int in_id, int out_id);

  void function_sum_exec( JitFunction function, 
			  int size, int threads, int blocks, int shared_mem_usage,
			  int in_id, int out_id);


  void function_bool_reduction_exec( JitFunction function, 
				     int size, int threads, int blocks, int shared_mem_usage,
				     int in_id, int out_id);

  

  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout >
  JitFunction 
  function_sum_convert_ind_build()
  {
    if (ptx_db::db_enabled) {
      JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (!func.empty())
	return func;
    }

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

    llvm_derefParam( p_lo ); // r_lo
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<T2WT>::value );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    llvm::Value* r_idx_perm = llvm_array_type_indirection( p_site_perm , r_idx );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   
    reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );

    sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)

    llvm_bar_sync();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();
    
    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T2>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  sdata_jit += sdata_reg_plus;
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    llvm::BasicBlock * block_store_global = llvm_new_basic_block();
    llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();
    llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
		      block_store_global , 
		      block_not_store_global );
    llvm_set_insert_point(block_store_global);
    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_sum_ind.ptx" , __PRETTY_FUNCTION__ );
  }



  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout >
  JitFunction 
  function_sum_convert_build()
  {
    if (ptx_db::db_enabled) {
      JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (!func.empty())
	return func;
    }

    llvm_start_new_function();

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo ); // r_lo
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<T2WT>::value );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
    //reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );
    reg_idata_elem.setup( idata.elem( input_layout , r_idx ) );

    sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)

    llvm_bar_sync();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();
    
    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T2>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  sdata_jit += sdata_reg_plus;
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    llvm::BasicBlock * block_store_global = llvm_new_basic_block();
    llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();
    llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
		      block_store_global , 
		      block_not_store_global );
    llvm_set_insert_point(block_store_global);
    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_sum_ind.ptx" , __PRETTY_FUNCTION__ );
  }




  template<class T1>
  JitFunction 
  function_sum_build()
  {
    if (ptx_db::db_enabled) {
      JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (!func.empty())
	return func;
    }

    llvm_start_new_function();

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t WT;

    ParamRef p_idata      = llvm_add_param< WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<WT>::value );


    typedef typename JITType<T1>::Type_t T1JIT;

    llvm::Value* r_idx = llvm_thread_idx();   

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_idx ) ); 

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later

    T1JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );

    zero_rep( sdata_jit );

    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    sdata_jit = reg_idata_elem; // 

    llvm_bar_sync();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();

    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T1>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  sdata_jit += sdata_reg_plus;
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    
    llvm::BasicBlock * block_store_global = llvm_new_basic_block();
    llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();
    llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
		      block_store_global , 
		      block_not_store_global );
    llvm_set_insert_point(block_store_global);
    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_sum.ptx" , __PRETTY_FUNCTION__ );
  }




  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout , class ConvertOp, class ReductionOp >
  JitFunction 
  function_bool_reduction_convert_build()
  {
    if (ptx_db::db_enabled) {
      JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (!func.empty())
	return func;
    }

    llvm_start_new_function();

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo ); // r_lo
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<T2WT>::value );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
    T2JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );

    ReductionOp::initNeutral( sdata_jit );
    
    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
    //reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );
    reg_idata_elem.setup( idata.elem( input_layout , r_idx ) );

    ConvertOp::apply( sdata_jit , reg_idata_elem );

    llvm_bar_sync();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();
    
    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T2>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  ReductionOp::apply( sdata_jit , sdata_reg_plus );
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    llvm::BasicBlock * block_store_global = llvm_new_basic_block();
    llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();
    llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
		      block_store_global , 
		      block_not_store_global );
    llvm_set_insert_point(block_store_global);
    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_sum_ind.ptx" , __PRETTY_FUNCTION__ );
  }

  


  template<class T1, class ReductionOp>
  JitFunction 
  function_bool_reduction_build()
  {
    if (ptx_db::db_enabled) {
      JitFunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (!func.empty())
	return func;
    }

    llvm_start_new_function();

    ParamRef p_lo     = llvm_add_param<int>();
    ParamRef p_hi     = llvm_add_param<int>();

    typedef typename WordType<T1>::Type_t WT;

    ParamRef p_idata      = llvm_add_param< WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T1>::Type_t> odata(  p_odata );   // want scalar access later

    llvm_derefParam( p_lo );
    llvm::Value* r_hi     = llvm_derefParam( p_hi );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<WT>::value );


    typedef typename JITType<T1>::Type_t T1JIT;

    llvm::Value* r_idx = llvm_thread_idx();   

    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // needed later

    typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;   
    reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_idx ) ); 

    IndexDomainVector args;
    args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later

    T1JIT sdata_jit;
    sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
    
    ReductionOp::initNeutral( sdata_jit );
    
    llvm_cond_exit( llvm_ge( r_idx , r_hi ) );

    sdata_jit = reg_idata_elem; // This is just copying booleans

    llvm_bar_sync();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();

    llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );


    //
    // Reduction loop
    //
    JitForLoopPower loop( r_pow_shr1 );
    {
      JitIf ifInRange( llvm_lt( r_tidx , loop.index() ) );
      {
	llvm::Value * v = llvm_add( loop.index() , r_tidx );

	JitIf ifInRange2( llvm_lt( v , r_ntidx ) );
	{
	  IndexDomainVector args_new;
	  args_new.push_back( make_pair( Layout::sitesOnNode() , 
					 llvm_add( r_tidx , loop.index() ) ) );  // sitesOnNode irrelevant since Scalar access later

	  typename JITType<T1>::Type_t sdata_jit_plus;
	  sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	  typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg_plus;
	  sdata_reg_plus.setup( sdata_jit_plus );

	  ReductionOp::apply( sdata_jit , sdata_reg_plus );
	}
	ifInRange2.end();
      }
      ifInRange.end();

      llvm_bar_sync();
    }
    loop.end();
    //
    // -------------------
    //

    
    llvm::BasicBlock * block_store_global = llvm_new_basic_block();
    llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();
    llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
		      block_store_global , 
		      block_not_store_global );
    llvm_set_insert_point(block_store_global);
    typename REGType< typename JITType<T1>::Type_t >::Type_t sdata_reg;   
    sdata_reg.setup( sdata_jit );
    odata.elem( JitDeviceLayout::Scalar , r_block_idx ) = sdata_reg;
    llvm_branch( block_not_store_global );
    llvm_set_insert_point(block_not_store_global);

    return jit_function_epilogue_get_cuf("jit_sum.ptx" , __PRETTY_FUNCTION__ );
  }


  

} // namespace

#endif
