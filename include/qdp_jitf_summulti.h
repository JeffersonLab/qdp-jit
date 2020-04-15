#ifndef QDP_JITF_SUMMULTI_H
#define QDP_JITF_SUMMULTI_H


namespace QDP {


  void
  function_summulti_convert_ind_exec( CUfunction function, 
				      int size, int threads, int blocks, int shared_mem_usage,
				      int in_id, int out_id,
				      int numsubsets,
				      const multi1d<int>& sizes,
				      const multi1d<QDPCache::ArgKey>& table_ids );

  void
  function_summulti_exec( CUfunction function, 
			  int size, int threads, int blocks, int shared_mem_usage,
			  int in_id, int out_id,
			  int numsubsets,
			  const multi1d<int>& sizes );


  
  // T1 input
  // T2 output
  template< class T1 , class T2 , JitDeviceLayout input_layout >
  CUfunction
  function_summulti_convert_ind_build()
  {
    /* if (ptx_db::db_enabled) { */
    /*   CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ ); */
    /*   if (func) */
    /* 	return func; */
    /* } */

    llvm_start_new_function();

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

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<T2WT>::value );

    typedef typename JITType<T2>::Type_t T2JIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_nblock_idx = llvm_call_special_nctaidx();
    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx      = llvm_call_special_ntidx(); // this is a power of 2


    llvm::BasicBlock * block_subset_loop_start = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_body  = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_body_cont1  = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_body_cont2  = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_exit  = llvm_new_basic_block();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();
    
    llvm_branch( block_subset_loop_start );

    llvm_set_insert_point( block_subset_loop_start );

    llvm::PHINode * r_subset = llvm_phi( llvm_type<int>::value , 2 );
    r_subset->addIncoming( llvm_create_value(0) , entry_block );

    llvm_cond_branch( llvm_ge( r_subset , r_subsetnum ) , block_subset_loop_exit , block_subset_loop_body );
    {
      llvm_set_insert_point(block_subset_loop_body);

      llvm_bar_sync();  // make sure thread block is synchronized
      
      llvm::Value* r_subset_inc = llvm_add( r_subset , llvm_create_value(1) );

      llvm::BasicBlock * block_store_global = llvm_new_basic_block();
      llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();

      r_subset->addIncoming( r_subset_inc , block_not_store_global );  // change this

      
      // Loop body begin: r_subset
      //

      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
      T2JIT sdata_jit;
      sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
      zero_rep( sdata_jit );

      llvm::Value* r_size = llvm_array_type_indirection( p_sizes , r_subset );

      llvm::BasicBlock * block_red_loop_end = llvm_new_basic_block();
      
      llvm_cond_branch( llvm_ge( r_idx , r_size ) , block_subset_loop_body_cont2 , block_subset_loop_body_cont1 ); //block_subset_loop_inc , block_not_store_global
      llvm_set_insert_point(block_subset_loop_body_cont1 );
      
      llvm::Value* r_sitetable = llvm_array_type_indirection( p_sitetables , r_subset );
      llvm::Value* r_idx_perm  = llvm_array_type_indirection( r_sitetable , r_idx );

      typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
      reg_idata_elem.setup( idata.elem( input_layout , r_idx_perm ) );

      sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)

      llvm_branch( block_subset_loop_body_cont2 );
      llvm_set_insert_point(block_subset_loop_body_cont2 );
      
      llvm_bar_sync(); // all threads need to execute this, otherwise leads to undefined behavior

      llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

      //
      // Shared memory reduction loop
      //
      llvm::BasicBlock * block_red_loop_start = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_start_1 = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_start_2 = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_add = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_sync = llvm_new_basic_block();

      llvm_branch( block_red_loop_start );
      llvm_set_insert_point(block_red_loop_start);
    
      llvm::PHINode * r_red_pow = llvm_phi( llvm_type<int>::value , 2 );
      r_red_pow->addIncoming( r_pow_shr1 , block_subset_loop_body_cont2 );  //block_power_loop_exit
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


      llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
			block_store_global , 
			block_not_store_global );
      llvm_set_insert_point(block_store_global);
      typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      
      llvm::Value* store_idx = llvm_add( llvm_mul( r_nblock_idx , r_subset ) , r_block_idx ); //   store:   subset * nblock  +  block
	
      odata.elem( JitDeviceLayout::Scalar , store_idx ) = sdata_reg;
      llvm_branch( block_not_store_global );
      llvm_set_insert_point(block_not_store_global);

      llvm_bar_sync();  // make sure thread block is synchronized
      
      llvm_branch( block_subset_loop_start );
    }

    llvm_set_insert_point(block_subset_loop_exit);

    
    return jit_function_epilogue_get_cuf("jit_summulti_ind.ptx" , __PRETTY_FUNCTION__ );
  }




  // T input/output
  template< class T >
  CUfunction
  function_summulti_build()
  {
    /* if (ptx_db::db_enabled) { */
    /*   CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ ); */
    /*   if (func) */
    /* 	return func; */
    /* } */

    llvm_start_new_function();

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

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_type<TWT>::value );

    typedef typename JITType<T>::Type_t TJIT;

    llvm::Value* r_idx = llvm_thread_idx();

    llvm::Value* r_nblock_idx = llvm_call_special_nctaidx();
    llvm::Value* r_block_idx  = llvm_call_special_ctaidx();
    llvm::Value* r_tidx       = llvm_call_special_tidx();
    llvm::Value* r_ntidx       = llvm_call_special_ntidx(); // this is a power of 2


    llvm::BasicBlock * block_subset_loop_start = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_body  = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_body_cont1  = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_body_cont2  = llvm_new_basic_block();
    llvm::BasicBlock * block_subset_loop_exit  = llvm_new_basic_block();

    llvm::BasicBlock * entry_block = llvm_get_insert_block();
    
    llvm_branch( block_subset_loop_start );

    llvm_set_insert_point( block_subset_loop_start );

    llvm::PHINode * r_subset = llvm_phi( llvm_type<int>::value , 2 );
    r_subset->addIncoming( llvm_create_value(0) , entry_block );

    llvm_cond_branch( llvm_ge( r_subset , r_subsetnum ) , block_subset_loop_exit , block_subset_loop_body );
    {
      llvm_set_insert_point(block_subset_loop_body);
      
      llvm_bar_sync();  // make sure thread block is synchronized
      
      llvm::Value* r_subset_inc = llvm_add( r_subset , llvm_create_value(1) );

      llvm::BasicBlock * block_store_global = llvm_new_basic_block();
      llvm::BasicBlock * block_not_store_global = llvm_new_basic_block();

      r_subset->addIncoming( r_subset_inc , block_not_store_global );  // change this

      
      // Loop body begin: r_subset
      //
      llvm::Value* r_size = llvm_array_type_indirection( p_sizes , r_subset );

      
      IndexDomainVector args;
      args.push_back( make_pair( Layout::sitesOnNode() , r_tidx ) );  // sitesOnNode irrelevant since Scalar access later
      TJIT sdata_jit;
      sdata_jit.setup( r_shared , JitDeviceLayout::Scalar , args );
      zero_rep( sdata_jit );

      llvm::BasicBlock * block_red_loop_end = llvm_new_basic_block();

      llvm_cond_branch( llvm_ge( r_idx , r_size ) , block_subset_loop_body_cont2 , block_subset_loop_body_cont1 ); //block_subset_loop_inc , block_not_store_global
      llvm_set_insert_point(block_subset_loop_body_cont1 );

      llvm::Value* r_in_idx = llvm_add( llvm_mul( r_subset , r_size ) , r_idx );

      typename REGType< typename JITType<T>::Type_t >::Type_t reg_idata_elem;
      reg_idata_elem.setup( idata.elem( JitDeviceLayout::Scalar , r_in_idx ) );

      sdata_jit = reg_idata_elem; // This should do the precision conversion (SP->DP)
      
      llvm_branch( block_subset_loop_body_cont2 );
      llvm_set_insert_point(block_subset_loop_body_cont2 );

      llvm_bar_sync();

      llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

      //
      // Shared memory reduction loop
      //
      llvm::BasicBlock * block_red_loop_start = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_start_1 = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_start_2 = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_add = llvm_new_basic_block();
      llvm::BasicBlock * block_red_loop_sync = llvm_new_basic_block();

      llvm_branch( block_red_loop_start );
      llvm_set_insert_point(block_red_loop_start);
    
      llvm::PHINode * r_red_pow = llvm_phi( llvm_type<int>::value , 2 );    
      r_red_pow->addIncoming( r_pow_shr1 , block_subset_loop_body_cont2 ); // block_power_loop_exit
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

      typename JITType<T>::Type_t sdata_jit_plus;
      sdata_jit_plus.setup( r_shared , JitDeviceLayout::Scalar , args_new );

      typename REGType< typename JITType<T>::Type_t >::Type_t sdata_reg_plus;    // 
      sdata_reg_plus.setup( sdata_jit_plus );

      sdata_jit += sdata_reg_plus;


      llvm_branch( block_red_loop_sync );

      llvm_set_insert_point(block_red_loop_sync);
      llvm_bar_sync();
      llvm::Value* pow_1 = llvm_shr( r_red_pow , llvm_create_value(1) );
      r_red_pow->addIncoming( pow_1 , block_red_loop_sync );

      llvm_branch( block_red_loop_start );

      llvm_set_insert_point(block_red_loop_end);


      llvm_cond_branch( llvm_eq( r_tidx , llvm_create_value(0) ) , 
			block_store_global , 
			block_not_store_global );
      llvm_set_insert_point(block_store_global);
      typename REGType< typename JITType<T>::Type_t >::Type_t sdata_reg;   
      sdata_reg.setup( sdata_jit );
      
      llvm::Value* store_idx = llvm_add( llvm_mul( r_nblock_idx , r_subset ) , r_block_idx ); //   store:   subset * nblock  +  block
	
      odata.elem( JitDeviceLayout::Scalar , store_idx ) = sdata_reg;
      llvm_branch( block_not_store_global );
      llvm_set_insert_point(block_not_store_global);
      
      llvm_bar_sync();  // make sure thread block is synchronized

      llvm_branch( block_subset_loop_start );
    }

    llvm_set_insert_point(block_subset_loop_exit);

    
    return jit_function_epilogue_get_cuf("jit_summulti_ind.ptx" , __PRETTY_FUNCTION__ );
  }



  

} // QDP

#endif
