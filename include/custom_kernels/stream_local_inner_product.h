#ifndef inline_prop_and_matelem_distillation_nvptx_h
#define inline_prop_and_matelem_distillation_nvptx_h

namespace QDP
{


  void function_multi_localInnerProduct_sum_convert_exec( JitFunction& function,
							  int size, 
							  int threads, 
							  int blocks, 
							  int shared_mem_usage,
							  multi1d<QDPCache::ArgKey>& in_ids, 
							  int out_id,
							  int v_id,
							  int N,
							  const multi1d<int>& sizes,
							  const multi1d<QDPCache::ArgKey>& table_ids);

    

  // T1 input
  // T2 output
  // T3 QDPType for localInnerProduct = vector
  template< class T1 , class T2 , class T3 , JitDeviceLayout input_layout >
  void
  function_multi_localInnerProduct_sum_convert_build(JitFunction& function)
  {
    llvm_start_new_function("multi_localInnerProduct_sum_convert",__PRETTY_FUNCTION__ );

    typedef typename WordType<T1>::Type_t T1WT;
    typedef typename WordType<T2>::Type_t T2WT;
    typedef typename WordType<T3>::Type_t T3WT;

    ParamRef p_numsubset  = llvm_add_param< int  >();   // number of subsets
    ParamRef p_sizes      = llvm_add_param< int* >();   // size (per subset)
    ParamRef p_sitetables = llvm_add_param< int** >();  // sitetable (per subset)
    ParamRef p_idata      = llvm_add_param< T1WT** >(); // Input  array (per subset)
    ParamRef p_odata      = llvm_add_param< T2WT* >();  // Output array 
    ParamRef p_vdata      = llvm_add_param< T3WT* >();  // Vector array

    OLatticeJIT<typename JITType<T1>::Type_t> idata(  p_idata );   // want scalar access later (since SubLat)
    OLatticeJIT<typename JITType<T2>::Type_t> odata(  p_odata );   // want scalar access later
    OLatticeJIT<typename JITType<T3>::Type_t> vdata(  p_vdata );   // want coal access later (since Lat)

    llvm::Value* r_subsetnum = llvm_derefParam( p_numsubset );

    llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<T2WT>() );

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

      //llvm::BasicBlock * block_red_loop_end = llvm_new_basic_block();
      
      JitIf if_init_shared( llvm_lt( r_idx , r_size ) );
      {
	llvm::Value* r_sitetable = llvm_array_type_indirection( p_sitetables , loop_subset.index() );
	llvm::Value* r_idx_perm  = llvm_array_type_indirection( r_sitetable , r_idx );

	ParamLeaf param_leaf;
	FnLocalInnerProduct op;
	auto op_jit = AddOpParam<FnLocalInnerProduct,ParamLeaf>::apply(op,param_leaf);

	typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
	reg_idata_elem.setup( idata.elem( input_layout , r_idx , loop_subset.index() ) );

	typename REGType< typename JITType<T3>::Type_t >::Type_t reg_vdata_elem;
	reg_vdata_elem.setup( vdata.elem( JitDeviceLayout::Coalesced , r_idx_perm ) );

	sdata_jit = op_jit( reg_idata_elem , reg_vdata_elem ); // This should do the precision conversion (SP->DP)
      }
      if_init_shared.end();
      
      llvm_bar_sync();

      llvm::Value* r_pow_shr1 = llvm_shr( r_ntidx , llvm_create_value(1) );

      //
      // Shared memory reduction loop
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

	    typename REGType< typename JITType<T2>::Type_t >::Type_t sdata_reg_plus;    // 
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

      // make sure thread block is synchronized
      llvm_bar_sync();  
    }
    loop_subset.end();


    jit_get_function(function);
  }




  template < class T1 , class T2 , class T3, JitDeviceLayout input_layout >
  void qdp_jit_multi_localInnerProduct_reduce_convert(int size, 
						      int threads, 
						      int blocks, 
						      int shared_mem_usage,
						      multi1d<QDPCache::ArgKey>& in_ids, 
						      int out_id,
						      int v_id,
						      int N,
						      const multi1d<int>& sizes,
						      const multi1d<QDPCache::ArgKey>& table_ids)
  {
    static JitFunction function;

    if (function.empty())
      function_multi_localInnerProduct_sum_convert_build<T1,T2,T3,input_layout>(function);

    function_multi_localInnerProduct_sum_convert_exec(function, size, threads, blocks, shared_mem_usage, in_ids, out_id, v_id , N, sizes, table_ids );
  }

  


  
  template<class T, class T3>
  void
  multi_innerProduct( multi1d< ComplexD* >& ret , const multi1d< OSubLattice<T>* >& ms1 , const OLattice<T3>& v1 )
  {
    const int N = ms1.size();
    
    //QDPIO::cout << "multi_innerProduct (GPU) with N = " << N << "\n";

    assert( N > 0 );
    assert( ret.size() == N );

    for (int i = 0 ; i < N ; ++i )
      {
	if (!ms1[i]->getOwnsMemory())
	  QDP_error_exit("sum with subtype view called");
      }


    typedef typename BinaryReturn<T,T3,FnLocalInnerProduct>::Type_t       TT3;
    typedef typename UnaryReturn<OLattice<TT3>, FnSum>::Type_t::SubType_t T2;

    typename UnaryReturn<OLattice<TT3>, FnSumMulti>::Type_t  dest( N );

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(T2) * N , dest.slice() , nullptr );

    
    multi1d<QDPCache::ArgKey> ms1_ids(N);
    multi1d<QDPCache::ArgKey> table_ids( N );
    multi1d<int>              sizes    ( N );

    // Zero-out the result (in case of empty subsets on the node)
    zero_rep( dest );
    
    for (int i = 0 ; i < N ; ++i ) {
      ms1_ids[i]  = QDPCache::ArgKey( ms1[i]->getId() );
      table_ids[i] = QDPCache::ArgKey( ms1[i]->subset().getIdSiteTable() );
      sizes[i]     = ms1[i]->subset().numSiteTable();
    }

    bool first=true;
    bool allocated=false;
    int out_id,in_id;

    while (1) {

      //QDPIO::cout << "sizes = ";      
      int maxsize = 0;
      for (int i = 0 ; i < N ; ++i )
	{
	  //QDPIO::cout << sizes[i] << " ";
	  if ( sizes[i] > maxsize )
	    maxsize = sizes[i];
	}
      if (maxsize == 0)
	break;
      //QDPIO::cout << "   ";
      //QDPIO::cout << "maxsize: " << maxsize << "\n";

      unsigned numThreads = gpu_getMaxBlockX();
      while ((numThreads*sizeof(T2) > gpu_getMaxSMem()) || (numThreads > (unsigned)maxsize)) {
	numThreads >>= 1;
      }
      unsigned numBlocks=(int)ceil(float(maxsize)/numThreads);

      if (numBlocks > gpu_getMaxGridX()) {
	QDP_error_exit( "sumMulti(Lat,set) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)gpu_getMaxGridX());
      }

      int shared_mem_usage = numThreads*sizeof(T2);
      //QDP_info("multi_innerProductsum(): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );

      if (first) {
	allocated=true;
	out_id = QDP_get_global_cache().add( numBlocks*sizeof(T2)*N , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	in_id  = QDP_get_global_cache().add( numBlocks*sizeof(T2)*N , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
      }

      
      if (numBlocks == 1)
	{
	  if (first)
	    {
	      qdp_jit_multi_localInnerProduct_reduce_convert<T,T2,T3,JitDeviceLayout::Scalar>(maxsize, numThreads, numBlocks, shared_mem_usage ,  // ok: Scalar
	      										      ms1_ids ,
	      										      d_id ,
	      										      v1.getId() ,
	      										      N, sizes, table_ids );
	    }
	  else
	    {
	      qdp_jit_summulti<T2>(maxsize, numThreads, numBlocks,
				   shared_mem_usage,
				   in_id, d_id,
				   N,
				   sizes);
	    }
	}
      else
	{
	  if (first)
	    {
	      qdp_jit_multi_localInnerProduct_reduce_convert<T,T2,T3,JitDeviceLayout::Scalar>(maxsize, numThreads, numBlocks, shared_mem_usage,       // ok: Scalar
											      ms1_ids,
											      out_id ,
											      v1.getId() ,
											      N, sizes, table_ids);
	    }
	  else
	    {
	      qdp_jit_summulti<T2>(maxsize, numThreads, numBlocks,
				   shared_mem_usage,
				   in_id, out_id,
				   N,
				   sizes);
	    }
	}

      first =false;

      if (numBlocks==1)
	break;

      //QDPIO::cout << "new sizes = ";
      for (int i = 0 ; i < N ; ++i )
	{
	  sizes[i] = numBlocks;
	  //QDPIO::cout << sizes[i] << " ";
	}
      //QDPIO::cout << "\n";

      int tmp = in_id;
      in_id = out_id;
      out_id = tmp;
    
    }

    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    // Global sum
    QDPInternal::globalSumArray(dest);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );

    
    if (allocated)
      {
	QDP_get_global_cache().signoff( in_id );
	QDP_get_global_cache().signoff( out_id );
      }
    
    for (int i = 0 ; i < N ; ++i )
      {
	*ret[i] = dest[i];
      }
    
  }

  


  
} // QDP


#endif
