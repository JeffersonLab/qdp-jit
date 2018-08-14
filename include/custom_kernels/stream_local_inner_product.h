#ifndef inline_prop_and_matelem_distillation_nvptx_h
#define inline_prop_and_matelem_distillation_nvptx_h

namespace QDP
{


  void function_multi_localInnerProduct_sum_convert_exec( CUfunction function,
							  int size, 
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
    assert( (threads & (threads - 1)) == 0 );

    int sizes_id = QDP_get_global_cache().add( sizes.size()*sizeof(int) , QDPCache::Flags::OwnHostMemory , QDPCache::Status::host , sizes.slice() , NULL , NULL );

    JitParam jit_numsubsets( QDP_get_global_cache().addJitParamInt( N ) );
    JitParam jit_tables(     QDP_get_global_cache().addMulti(       table_ids  ) );
    JitParam jit_in_ids(     QDP_get_global_cache().addMulti(       in_ids ) );
	      
    std::vector<QDPCache::ArgKey> ids;
    ids.push_back( jit_numsubsets.get_id() );
    ids.push_back( sizes_id );
    ids.push_back( jit_tables.get_id() );
    ids.push_back( jit_in_ids.get_id() );
    ids.push_back( out_id );
    ids.push_back( v_id );

    std::vector<void*> args( QDP_get_global_cache().get_kernel_args(ids) );
    kernel_geom_t now = getGeom( size , threads );

    CudaLaunchKernel(function,   now.Nblock_x,now.Nblock_y,1,    threads,1,1,    shared_mem_usage, 0, &args[0] , 0);

    QDP_get_global_cache().signoff(sizes_id);
  }

    

  // T1 input
  // T2 output
  // T3 QDPType for localInnerProduct = vector
  template< class T1 , class T2 , class T3 , JitDeviceLayout input_layout >
  CUfunction 
  function_multi_localInnerProduct_sum_convert_build()
  {
    if (ptx_db::db_enabled) {
      CUfunction func = llvm_ptx_db( __PRETTY_FUNCTION__ );
      if (func)
	return func;
    }

    llvm_start_new_function();

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
      
      llvm_cond_branch( llvm_ge( r_idx , r_size ) , block_red_loop_end , block_subset_loop_body_cont1 ); //block_subset_loop_inc , block_not_store_global
      llvm_set_insert_point(block_subset_loop_body_cont1 );
      
      llvm::Value* r_sitetable = llvm_array_type_indirection( p_sitetables , r_subset );
      llvm::Value* r_idx_perm  = llvm_array_type_indirection( r_sitetable , r_idx );


      ParamLeaf param_leaf;
      FnLocalInnerProduct op;
      auto op_jit = AddOpParam<FnLocalInnerProduct,ParamLeaf>::apply(op,param_leaf);

      typename REGType< typename JITType<T1>::Type_t >::Type_t reg_idata_elem;
      reg_idata_elem.setup( idata.elem( input_layout , r_idx , r_subset ) );

      typename REGType< typename JITType<T3>::Type_t >::Type_t reg_vdata_elem;
      reg_vdata_elem.setup( vdata.elem( JitDeviceLayout::Coalesced , r_idx_perm ) );

      sdata_jit = op_jit( reg_idata_elem , reg_vdata_elem ); // This should do the precision conversion (SP->DP)

      
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
      r_red_pow->addIncoming( r_pow_shr1 , block_subset_loop_body_cont1 );  //block_power_loop_exit
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

    
    return jit_function_epilogue_get_cuf("jit_multi_innerproduct.ptx" , __PRETTY_FUNCTION__ );
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
    static CUfunction function;

    if (function == NULL)
      function = function_multi_localInnerProduct_sum_convert_build<T1,T2,T3,input_layout>();

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

    multi1d<QDPCache::ArgKey> ms1_ids(N);
    multi1d<QDPCache::ArgKey> table_ids( N );
    multi1d<int>              sizes    ( N );
    
    for (int i = 0 ; i < N ; ++i ) {
      zero_rep( dest[i] );
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

      unsigned numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || (numThreads > (unsigned)maxsize)) {
	numThreads >>= 1;
      }
      unsigned numBlocks=(int)ceil(float(maxsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_error_exit( "sumMulti(Lat,set) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)DeviceParams::Instance().getMaxGridX());
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
	      										      dest.getId() ,
	      										      v1.getId() ,
	      										      N, sizes, table_ids );
	    }
	  else
	    {
	      qdp_jit_summulti<T2>(maxsize, numThreads, numBlocks,
				   shared_mem_usage,
				   in_id, dest.getId(),
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

    QDPInternal::globalSumArray(dest);

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
