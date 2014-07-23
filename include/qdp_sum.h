#ifndef SUM_H
#define SUM_H

namespace QDP {


  template <class T2>
  void reduce_convert(int size, int threads, int blocks, int shared_mem_usage,
		      T2 *d_idata, T2 *d_odata )
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

    function_sum_exec(function, size, threads, blocks, shared_mem_usage, (void*)d_idata, (void*)d_odata );
  }


  // T1 input
  // T2 output
  template < class T1 , class T2 , JitDeviceLayout input_layout >
  void reduce_convert_indirection(int size, 
				  int threads, 
				  int blocks, 
				  int shared_mem_usage,
				  T1 *d_idata, 
				  T2 *d_odata, 
				  int * siteTable)
  {
    static CUfunction function;

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_sum_ind_build<T1,T2,input_layout>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    // Execute the function
    function_sum_ind_exec(function, size, threads, blocks, shared_mem_usage, 
			  (void*)d_idata, (void*)d_odata, (void*)siteTable );
  }





  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSum>::Type_t
  sum(const OLattice<T1>& s1, const Subset& s)
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;
    
    //QDP_info("sum(lat,subset) dev");

    T2 * out_dev;
    T2 * in_dev;

    typename UnaryReturn<OLattice<T1>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
    static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
    prof.stime(getClockTime());
#endif

    int actsize=s.numSiteTable();
    bool first=true;
    while (1) {

      int numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || (numThreads > actsize)) {
	numThreads >>= 1;
      }
      int numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_error_exit( "sum(Lat,subset) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)DeviceParams::Instance().getMaxGridX());
      }

      int shared_mem_usage = numThreads*sizeof(T2);
      //QDP_info("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );

      if (first) {
	if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat,subset) reduction buffer: 1st buffer no memory, exit");
	if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat,subset) reduction buffer: 2nd buffer no memory, exit");
      }

      if (numBlocks == 1) {
	if (first) {
	  reduce_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(actsize, numThreads, numBlocks, shared_mem_usage ,  
								       (T1*)QDPCache::Instance().getDevicePtr( s1.getId() ),
								       (T2*)QDPCache::Instance().getDevicePtr( d.getId() ),
								       (int*)QDPCache::Instance().getDevicePtr( s.getIdSiteTable()));
	}
	else {
	  reduce_convert<T2>( actsize , numThreads , numBlocks, shared_mem_usage , 
			      in_dev , (T2*)QDPCache::Instance().getDevicePtr( d.getId() ) );
	}
      } else {
	if (first) {
	  reduce_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(actsize, numThreads, numBlocks, shared_mem_usage,
								       (T1*)QDPCache::Instance().getDevicePtr( s1.getId() ),
								       out_dev , (int*)QDPCache::Instance().getDevicePtr(s.getIdSiteTable()));
	}
	else
	  reduce_convert<T2>( actsize , numThreads , numBlocks , shared_mem_usage , in_dev , out_dev );

      }

      first =false;

      if (numBlocks==1) 
	break;

      actsize=numBlocks;

      T2 * tmp = in_dev;
      in_dev = out_dev;
      out_dev = tmp;
    }

    QDPCache::Instance().free_device_static( in_dev );
    QDPCache::Instance().free_device_static( out_dev );

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
  void globalMax_kernel(int size, int threads, int blocks, T *d_idata, T *d_odata)
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

    function_global_max_exec(function, size, threads, blocks, shared_mem_usage, (void*)d_idata, (void*)d_odata );
  }





  //
  // sumMulti DOUBLE PRECISION
  //
  template<class T1>
    typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t
    sumMulti( const OLattice<T1>& s1 , const Set& ss )
    {
      typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t  dest(ss.numSubsets());
      const int nodeSites = Layout::sitesOnNode();

      while(1) {

	//
	// T2 is the upcasted version of T1
	// FnSum is okay to use here!
	//
	typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

	if (!ss.enableGPU) {
	  QDPIO::cout << "sumMulti called with a set, that is not supported for execution on the device\n";
	  break;
	}

	//QDPIO::cout << "sumMulti(Lat) dev, ss.largest_subset = " << ss.largest_subset << "\n";

	int numThreads = 
	  ss.largest_subset > DeviceParams::Instance().getMaxBlockX() ? 
	  DeviceParams::Instance().getMaxBlockX() : 
	  ss.largest_subset;

	bool no_way=false;

	while ( (numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || 
		(ss.largest_subset % numThreads) ) {
	  numThreads >>= 1;
	  if (numThreads == 0) {
	    no_way=true;
	    break;
	  }
	  /* string tmp = ss.largest_subset % numThreads ? "true":"false"; */
	  /* QDP_debug("numThreads=%d subset size mod=%s" , numThreads , tmp.c_str()); */
	}

	if (no_way) {
	  QDPIO::cout << "sumMulti: No suitable number of threads per blocks found. Largest subset size = " << ss.largest_subset << "\n";
	  break;
	}

	int numBlocks=(int)ceil(float(nodeSites)/numThreads);

	/* QDPIO::cout << "using " << numThreads  */
	/* 	    << " threads per block smem = " << numThreads*sizeof(T2)  */
	/* 	    << " numBlocks = " << numBlocks << "\n"; */

	if (numBlocks > DeviceParams::Instance().getMaxGridX() ) {
	  QDP_info_primary( "sum(Lat) numBlocks > maxGrixX, continue on host" );
	  break;
	}

	T2 * out_dev;
	T2 * in_dev;

	if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sumMulti(lat) reduction buffer: 2nd buffer no memory" );

	if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit("sumMulti(lat) reduction buffer: 3rd buffer no memory" );

	int virt_size = ss.largest_subset;

	int actsize=nodeSites;
	bool first=true;
	bool success=false;
	while (1) {

	  int shared_mem_usage = numThreads*sizeof(T2);

	  //QDP_info("numBlocks=%d actsize=%d virt_size=%d smem=%d",numBlocks,actsize,virt_size,shared_mem_usage);

	  if (first) {
	    reduce_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(actsize, 
									 numThreads, 
									 numBlocks,  
									 shared_mem_usage,
									 (T1*)QDPCache::Instance().getDevicePtr( s1.getId() ),
									 out_dev , 
									 (int*)QDPCache::Instance().getDevicePtr( ss.getIdStrided() ) );
	  } else {
	    reduce_convert<T2>(actsize, 
			       numThreads, 
			       numBlocks, 
			       shared_mem_usage,
			       in_dev, 
			       out_dev );
	  }

	  if (first) {
	    first =false;
	  }

	  T2 * tmp = in_dev;
	  in_dev = out_dev;
	  out_dev = tmp;
      
#ifdef GPU_DEBUG_DEEP
	  QDP_debug_deep( "checking for break numBlocks = %d %d" , numBlocks , ss.nonEmptySubsetsOnNode );
#endif
	  if ( numBlocks == ss.nonEmptySubsetsOnNode ) {
	    success=true;
	    break;
	  }

	  virt_size /= numThreads;

	  numThreads = virt_size > DeviceParams::Instance().getMaxBlockX() ? DeviceParams::Instance().getMaxBlockX() : virt_size;
	  actsize = numBlocks;
	  numBlocks=(int)ceil(float(actsize)/numThreads);

	  //QDP_info("numThreads=%d numBlocks=%d",numThreads,numBlocks);

	  no_way=false;
	  while ( (numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || 
		  (virt_size % numThreads) ) {
#ifdef GPU_DEBUG_DEEP
	    QDP_debug_deep( "loop entered %d %d %d" , numThreads*sizeof(T2) , virt_size , numThreads );
#endif
	    numThreads >>= 1;
	    if (numThreads == 0) {
	      no_way=true;
	      break;
	    }
	    numBlocks=(int)ceil(float(actsize)/numThreads);
/* 	    string tmp = virt_size % numThreads ? "true":"false"; */
/* #ifdef GPU_DEBUG_DEEP */
/* 	    QDP_debug_deep("numThreads=%d subset size mod=%s" , numThreads , tmp.c_str() ); */
/* #endif */
	  }
      
	  if (no_way) {
	    QDP_info_primary( "sumMulti: No number of threads per blocks found that suits the requirements. Largest subset size = %d",
		      ss.largest_subset);
	    break;
	  }

#ifdef GPU_DEBUG_DEEP
	  QDP_debug_deep("using %d threads per block smem=%d numBlocks=%d" , numThreads , numThreads*sizeof(T2) , numBlocks);
#endif
      
	}

	T2 * tmp = in_dev;
	in_dev = out_dev;
	out_dev = tmp;

	T2* slice = new T2[ss.numSubsets()];

	CudaMemcpyD2H( (void*)slice , (void*)out_dev , ss.numSubsets()*sizeof(T2) );

	QDPCache::Instance().free_device_static( in_dev );
	QDPCache::Instance().free_device_static( out_dev );
    
	if (!success) {
	  QDP_info_primary("sumMulti: there was a problem, continue on host");
	  delete[] slice;
	  break;
	}

	for (int i = 0 ; i < ss.numSubsets() ; ++i ) {
	  zero_rep(dest[i].elem());
	}

	//QDP_debug("ss.stride_offset = %d ss.nonEmptySubsetsOnNode = %d" ,ss.stride_offset, ss.nonEmptySubsetsOnNode);

	//
	// "dest" and "slice" are both on DP
	//
	for (int i = 0 ; i < ss.nonEmptySubsetsOnNode ; ++i ) {
	  dest[ ss.stride_offset + i ].elem() = slice[i];
	}
    
	delete[] slice;

	//
	// We need to unlock things, so that the global sum can be carried out
	//
    
	QDPInternal::globalSumArray(dest);

	return dest;
      }

      QDPIO::cout << "sumMulti on host\n";

#if defined(QDP_USE_PROFILING)   
	static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
	prof.time -= getClockTime();
#endif

	// Initialize result with zero
	for(int k=0; k < ss.numSubsets(); ++k)
	  zero_rep(dest[k]);

	// Loop over all sites and accumulate based on the coloring 
	const multi1d<int>& lat_color =  ss.latticeColoring();
	//const int nodeSites = Layout::sitesOnNode();

	for(int i=0; i < nodeSites; ++i) 
	  {
	    int j = lat_color[i];
	    dest[j].elem() += s1.elem(i);
	  }

	// Do a global sum on the result
	QDPInternal::globalSumArray(dest);

#if defined(QDP_USE_PROFILING)   
	prof.time += getClockTime();
	prof.count++;
	prof.print();
#endif

	return dest;
    }







      template<class T>
	typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
	globalMax(const OLattice<T>& s1)
	{
	  //    QDP_info("globalMax(lat) dev");

	  T * out_dev;
	  T * in_dev;
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
	      if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T) ))
		QDP_error_exit( "globMax(lat) reduction buffer: 1st buffer no memory, exit");
	      if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T) ))
		QDP_error_exit( "globMax(lat) reduction buffer: 2nd buffer no memory, exit");
	    }

	    if (numBlocks == 1) {
	      if (first)
		globalMax_kernel<T>(actsize, numThreads, numBlocks, 
				    (T*)QDPCache::Instance().getDevicePtr( s1.getId() ),
				    (T*)QDPCache::Instance().getDevicePtr( d.getId() ) );
	      else
		globalMax_kernel<T>(actsize, numThreads, numBlocks, 
				    in_dev, 
				    (T*)QDPCache::Instance().getDevicePtr( d.getId() ) );
	    } else {
	      if (first)
		globalMax_kernel<T>(actsize, numThreads, numBlocks, 
				    (T*)QDPCache::Instance().getDevicePtr( s1.getId() ),
				    out_dev );
	      else
		globalMax_kernel<T>(actsize, numThreads, numBlocks, 
				    in_dev, 
				    out_dev );
	    }

	    first =false;

	    if (numBlocks==1) 
	      break;

	    actsize=numBlocks;

	    T * tmp = in_dev;
	    in_dev = out_dev;
	    out_dev = tmp;
	  }

	  QDPCache::Instance().free_device_static( in_dev );
	  QDPCache::Instance().free_device_static( out_dev );

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
