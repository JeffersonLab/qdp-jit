#ifndef SUM_H
#define SUM_H

namespace QDP {


  template <class T2>
  void qdp_jit_reduce(int size, int threads, int blocks, int shared_mem_usage, int in_id, int out_id )
  {
    static CUfunction function;

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_sum_build<T2>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    function_sum_exec(function, size, threads, blocks, shared_mem_usage, in_id, out_id );
  }


  // T1 input
  // T2 output
  template < class T1 , class T2 , JitDeviceLayout input_layout >
  void qdp_jit_reduce_convert_indirection(int size, 
					  int threads, 
					  int blocks, 
					  int shared_mem_usage,
					  int in_id, 
					  int out_id, 
					  int siteTableId)
  {
    static CUfunction function;

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_sum_convert_ind_build<T1,T2,input_layout>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    // Execute the function
    function_sum_convert_ind_exec(function, size, threads, blocks, shared_mem_usage, 
				  in_id, out_id, siteTableId );
  }


  // T1 input
  // T2 output
  template < class T1 , class T2 , JitDeviceLayout input_layout >
  void qdp_jit_summulti_convert_indirection(int size, 
					    int threads, 
					    int blocks, 
					    int shared_mem_usage,
					    int in_id, 
					    int out_id,
					    int numsubsets,
					    const multi1d<int>& sizes,
					    const multi1d<int>& table_ids)
  {
    static CUfunction function;

    assert( sizes.size() == numsubsets );
    assert( table_ids.size() == numsubsets );

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_summulti_convert_ind_build<T1,T2,input_layout>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    // Execute the function
    function_summulti_convert_ind_exec(function,
				       size, threads, blocks, shared_mem_usage, 
				       in_id, out_id,
				       numsubsets ,
				       sizes ,
				       table_ids );
  }



  // T input/output
  template < class T >
  void qdp_jit_summulti(int size, 
			int threads, 
			int blocks, 
			int shared_mem_usage,
			int in_id, 
			int out_id,
			int numsubsets,
			const multi1d<int>& sizes)
  {
    static CUfunction function;

    assert( sizes.size() == numsubsets );

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_summulti_build<T>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    // Execute the function
    function_summulti_exec(function,
			   size, threads, blocks, shared_mem_usage, 
			   in_id, out_id,
			   numsubsets ,
			   sizes );
  }



  // T1 input
  // T2 output
  template < class T1 , class T2 , JitDeviceLayout input_layout >
  void qdp_jit_reduce_convert(int size, 
			      int threads, 
			      int blocks, 
			      int shared_mem_usage,
			      int in_id, 
			      int out_id)
  {
    static CUfunction function;

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_sum_convert_build<T1,T2,input_layout>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    // Execute the function
    function_sum_convert_exec(function, size, threads, blocks, shared_mem_usage, 
			      in_id, out_id );
  }





  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSum>::Type_t
  sum(const OLattice<T1>& s1, const Subset& s)
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;
    
    //QDP_info("sum(lat,subset) dev");

    int out_id,in_id;

    typename UnaryReturn<OLattice<T1>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
    static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
    prof.stime(getClockTime());
#endif

    unsigned actsize=s.numSiteTable();
    bool first=true;
    while (1) {

      unsigned numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || (numThreads > actsize)) {
	numThreads >>= 1;
      }
      unsigned numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_error_exit( "sum(Lat,subset) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)DeviceParams::Instance().getMaxGridX());
      }

      int shared_mem_usage = numThreads*sizeof(T2);
      //QDP_info("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );

      if (first) {
	out_id = QDP_get_global_cache().add( numBlocks*sizeof(T2) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	in_id  = QDP_get_global_cache().add( numBlocks*sizeof(T2) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
      }

      
      if (numBlocks == 1) {
	if (first) {
	  qdp_jit_reduce_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(actsize, numThreads, numBlocks, shared_mem_usage, s1.getId(), d.getId(), s.getIdSiteTable());
	}
	else {
	  qdp_jit_reduce<T2>( actsize , numThreads , numBlocks, shared_mem_usage , in_id , d.getId() );
	}
      } else {
	if (first) {
	  qdp_jit_reduce_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(actsize, numThreads, numBlocks, shared_mem_usage, s1.getId(), out_id, s.getIdSiteTable());
	}
	else
	  qdp_jit_reduce<T2>( actsize , numThreads , numBlocks , shared_mem_usage , in_id , out_id );

      }

      first =false;

      if (numBlocks==1) 
	break;

      actsize=numBlocks;

      int tmp = in_id;
      in_id = out_id;
      out_id = tmp;
    }

    QDP_get_global_cache().signoff( in_id );
    QDP_get_global_cache().signoff( out_id );

    QDPInternal::globalSum(d);

#if defined(QDP_USE_PROFILING)   
    prof.etime(getClockTime());
    prof.count++;
    prof.print();
#endif

    return d;
  }


  //
  // globalMax
  //
  template <class T>
  void globalMax_kernel(int size, int threads, int blocks, int in_id, int out_id)
  {
    int shared_mem_usage = threads * sizeof(T);

    static CUfunction function;

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_global_max_build<T>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    function_global_max_exec(function, size, threads, blocks, shared_mem_usage, in_id, out_id );
  }





  //
  // sumMulti DOUBLE PRECISION
  //
  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t
  sumMulti( const OLattice<T1>& s1 , const Set& ss )
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

    const int numsubsets = ss.numSubsets();
    
    int maxsize = 0;
    multi1d<int> table_ids( numsubsets );
    for (int i = 0 ; i < numsubsets ; ++i )
      {
	table_ids[i] = ss[i].getIdSiteTable();
	if ( ss[i].numSiteTable() > maxsize )
	  maxsize = ss[i].numSiteTable();
      }
    
    //QDPIO::cout << "number of subsets:           " << ss.numSubsets() << "\n";
      
    //QDPIO::cout << "sizes = ";
    multi1d<int> sizes(numsubsets);
    for (int i = 0 ; i < numsubsets ; ++i )
      {
	sizes[i] = ss[i].numSiteTable();
	//QDPIO::cout << sizes[i] << " ";
      }
    //QDPIO::cout << "\n";
    
    bool first=true;
    
    int out_id,in_id;
    
    while( 1 ) {

      int maxsize = 0;
      for (int i = 0 ; i < numsubsets ; ++i )
	{
	  if ( sizes[i] > maxsize )
	    maxsize = sizes[i];
	}
      
      //QDPIO::cout << "maxsize: " << maxsize << "\n";

      unsigned numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || (numThreads > maxsize)) {
	numThreads >>= 1;
      }
      unsigned numBlocks=(int)ceil(float(maxsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_error_exit( "sum(Lat,subset) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)DeviceParams::Instance().getMaxGridX());
      }

      int shared_mem_usage = numThreads*sizeof(T2);
      //QDP_info("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );

      if (first) {
	out_id = QDP_get_global_cache().add( numBlocks*sizeof(T2)*numsubsets , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	in_id  = QDP_get_global_cache().add( numBlocks*sizeof(T2)*numsubsets , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
      }

      if (numBlocks == 1) {
	if (first) {
	  qdp_jit_summulti_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(maxsize, numThreads, numBlocks,
										 shared_mem_usage,
										 s1.getId(), out_id,
										 numsubsets,
										 sizes,
										 table_ids);
	}
	else {
	  qdp_jit_summulti<T2>(maxsize, numThreads, numBlocks,
			       shared_mem_usage,
			       in_id, out_id,
			       numsubsets,
			       sizes);
	}
      } else {
	if (first) {
	      qdp_jit_summulti_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(maxsize, numThreads, numBlocks,
										     shared_mem_usage,
										     s1.getId(), out_id,
										     numsubsets,
										     sizes,
										     table_ids);
	}
	else {
	  qdp_jit_summulti<T2>(maxsize, numThreads, numBlocks,
			       shared_mem_usage,
			       in_id, out_id,
			       numsubsets,
			       sizes);
	}

      }

#if 0
      {
	multi1d<double> tmp(numBlocks*numsubsets);
	std::vector<int> ids = {out_id};
	auto ptrs = QDP_get_global_cache().get_kernel_args( ids , false );
      
	CudaMemcpyD2H( (void*)tmp.slice() , ptrs[0] , numBlocks*numsubsets*sizeof(double) );
	QDPIO::cout << "out: ";
	for(int i=0;i<tmp.size();i++)
	  QDPIO::cout << tmp[i] << " ";
	QDPIO::cout << "\n";
      }
#endif
      
      first =false;

      if (numBlocks==1)
	break;

      //QDPIO::cout << "new sizes = ";
      for (int i = 0 ; i < numsubsets ; ++i )
	{
	  sizes[i] = numBlocks;
	  //QDPIO::cout << sizes[i] << " ";
	}
      //QDPIO::cout << "\n";

      int tmp = in_id;
      in_id = out_id;
      out_id = tmp;
    }

    
    qdp_stack_scalars_start_from_id( out_id );

    typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t  dest(ss.numSubsets());

    qdp_stack_scalars_end();
    
    QDPInternal::globalSumArray(dest);
    
    QDP_get_global_cache().signoff( in_id );
    QDP_get_global_cache().signoff( out_id );
      
    return dest;
  }




  template<class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
  globalMax(const OLattice<T>& s1)
  {
    int out_id, in_id;
	  
    const int nodeSites = Layout::sitesOnNode();

    typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
    static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
    prof.stime(getClockTime());
#endif

    int actsize=nodeSites;
    bool first=true;
    while (1) {

      int numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T) > DeviceParams::Instance().getMaxSMem()) || (numThreads > actsize)) {
	numThreads >>= 1;
      }
      int numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_error_exit( "sum(Lat,subset) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)DeviceParams::Instance().getMaxGridX());
      }

      if (first) {
	out_id = QDP_get_global_cache().add( numBlocks*sizeof(T) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	in_id  = QDP_get_global_cache().add( numBlocks*sizeof(T) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
      }


      if (numBlocks == 1) {
	if (first)
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, s1.getId() , d.getId() );
	else
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, in_id, d.getId() );
      } else {
	if (first)
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, s1.getId() , out_id );
	else
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, in_id, out_id );
      }

      first =false;

      if (numBlocks==1) 
	break;

      actsize=numBlocks;

      int tmp = in_id;
      in_id = out_id;
      out_id = tmp;
    }

    QDP_get_global_cache().signoff( in_id );
    QDP_get_global_cache().signoff( out_id );

    QDPInternal::globalMax(d);

#if defined(QDP_USE_PROFILING)   
    prof.etime( getClockTime() );
    prof.count++;
    prof.print();
#endif

    return d;
  }


    }

#endif
