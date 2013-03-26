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



  template <class T1,class T2>
  void reduce_convert_indirection_coal(int size, int threads, int blocks, int shared_mem_usage,
				       T1 *d_idata, T2 *d_odata, int * siteTable)
  {
    static CUfunction function;

    // Build the function
    if (function == NULL)
      {
	//std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	function = function_sum_ind_coal_build<T1,T2>();
	//std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
      }
    else
      {
	//std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
      }

    // Execute the function
    function_sum_ind_coal_exec(function, size, threads, blocks, shared_mem_usage, 
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
	  reduce_convert_indirection_coal<T1,T2>(actsize, numThreads, numBlocks, shared_mem_usage ,  
						 (T1*)QDPCache::Instance().getDevicePtr( s1.getId() ),
						 (T2*)QDPCache::Instance().getDevicePtr( d.getId() ),
						 (int*)QDPCache::Instance().getDevicePtr( s.getId()) );
	}
	else {
	  reduce_convert<T2>( actsize , numThreads , numBlocks, shared_mem_usage , 
			      in_dev , (T2*)QDPCache::Instance().getDevicePtr( d.getId() ) );
	}
      } else {
	if (first) {
	  reduce_convert_indirection_coal<T1,T2>(actsize, numThreads, numBlocks, shared_mem_usage,
						 (T1*)QDPCache::Instance().getDevicePtr( s1.getId() ),
						 out_dev , (int*)QDPCache::Instance().getDevicePtr(s.getId()) );
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




  template<class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
  globalMax(const OLattice<T>& s1)
  {
    //    QDP_info("globalMax(lat) dev");

    T * out_dev;
    T * in_dev;
    const int nodeSites = Layout::sitesOnNode();

    typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

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

    return d;
  }


}

#endif
