#ifndef QDP_JITF_SUMMULTI_H
#define QDP_JITF_SUMMULTI_H


namespace QDP {


  void
  function_summulti_convert_ind_exec( JitFunction& function, 
				      int size, int threads, int blocks, 
				      int in_id, int out_id,
				      int numsubsets,
				      const multi1d<int>& sizes,
				      const multi1d<QDPCache::ArgKey>& table_ids );

  void
  function_summulti_exec( JitFunction& function, 
			  int size, int threads, int blocks, 
			  int in_id, int out_id,
			  int numsubsets,
			  const multi1d<int>& sizes );


  
  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout >
  void
  function_summulti_convert_ind_build(JitFunction& function)
  {
    llvm_start_new_function("summulti_convert_ind",__PRETTY_FUNCTION__ );

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;

    ParamRef p_numsubset  = llvm_add_param< int  >();   // number of subsets
    ParamRef p_sizes      = llvm_add_param< int* >();   // size (per subset)
    ParamRef p_sitetables = llvm_add_param< int** >();  // sitetable (per subset)
    ParamRef p_idata      = llvm_add_param< T1WT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // output array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want coal   access later
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later

    llvm::Value* r_subsetnum = llvm_derefParam( p_numsubset );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<T2WT>() , gpu_getMaxSMem() / sizeof(T2WT) );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_nblock_idx = llvm_call_special_nctaidx();
    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx      = llvm_call_special_ntidx(); // this is a power of 2


    JitForLoop loop_subset( 0 , r_subsetnum );
    {
      llvm_bar_sync();  // make sure thread block is synchronized
      
      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
      T2JIT sdata_jit;
      sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
      zero_rep( sdata_jit );

      llvm::Value* r_size = llvm_array_type_indirection( p_sizes , loop_subset.index() );


      JitIf ifInRange( llvm_lt( r_idx , r_size ) );
      {
	llvm::Value* r_sitetable = llvm_array_type_indirection( p_sitetables , loop_subset.index() );
	llvm::Value* r_idx_perm  = llvm_array_type_indirection( r_sitetable , r_idx );

	typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
	reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );

	sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)
      }
      ifInRange.end();

      
      llvm_bar_sync(); // all threads need to execute this, otherwise leads to undefined behavior


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
      
      JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
      {
	typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
	sdata_reg.setup( sdata_jit );
      
	llvm::Value* store_idx = llvm_add( llvm_mul( r_nblock_idx , loop_subset.index() ) , r_block_idx ); //   store:   subset * nblock  +  block
	
	odata.elem( JitDeviceLayout::Scalar , store_idx ) = sdata_reg;
      }
      ifStore.end();
      
      llvm_bar_sync();  // make sure thread block is synchronized
      
    }
    loop_subset.end();

    jit_get_function(function);
  }




  // T input/output
  template< class T >
  void
  function_summulti_build(JitFunction& function)
  {
    llvm_start_new_function("summulti",__PRETTY_FUNCTION__ );

    typedef typename WordType<T>::Type_t TWT;

    ParamRef p_numsubset  = llvm_add_param< int  >();   // number of subsets
    ParamRef p_sizes      = llvm_add_param< int* >();   // size (per subset)
    ParamRef p_idata      = llvm_add_param< TWT* >();  // Input  array
    ParamRef p_odata      = llvm_add_param< TWT* >();  // output array

    OLatticeJIT<typename JITType<T>::Type_t> idata(  p_idata );   // want scalar access later
    OLatticeJIT<typename JITType<T>::Type_t> odata(  p_odata );   // want scalar access later

    llvm::Value* r_subsetnum = llvm_derefParam( p_numsubset );

    llvm_derefParam( p_idata );  // Input  array
    llvm_derefParam( p_odata );  // output array

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<TWT>() , gpu_getMaxSMem() / sizeof(TWT) );

    typedef typename JITType<T>::Type_t TJIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_nblock_idx = llvm_call_special_nctaidx();
    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2


    JitForLoop loop_subset( 0 , r_subsetnum );
    {
      llvm_bar_sync();  // make sure thread block is synchronized
      
      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
      TJIT sdata_jit;
      sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
      zero_rep( sdata_jit );

      llvm::Value* r_size = llvm_array_type_indirection( p_sizes , loop_subset.index() );


      JitIf ifInRange( llvm_lt( r_idx , r_size ) );
      {
	llvm::Value* r_in_idx = llvm_add( llvm_mul( loop_subset.index() , r_size ) , r_idx );
	
	typename REGType< typename JITType<T>::Type_t >::Type_t reg_idata_elem;
	reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_in_idx ) );

	sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)
      }
      ifInRange.end();

      
      llvm_bar_sync(); // all threads need to execute this, otherwise leads to undefined behavior


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

	    typename JITType<T>::Type_t sdata_jit_plus;
	    sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

	    typename REGType< typename JITType<T>::Type_t >::Type_t sdata_reg_plus;
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
      
      JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
      {
	typename REGType< typename JITType<T>::Type_t >::Type_t sdata_reg;   
	sdata_reg.setup( sdata_jit );
      
	llvm::Value* store_idx = llvm_add( llvm_mul( r_nblock_idx , loop_subset.index() ) , r_block_idx ); //   store:   subset * nblock  +  block
	
	odata.elem( JitDeviceLayout::Scalar , store_idx ) = sdata_reg;
      }
      ifStore.end();
      
      llvm_bar_sync();  // make sure thread block is synchronized
      
    }
    loop_subset.end();

    
    jit_get_function(function);
  }



  

} // QDP

#endif
