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

  //  exit(1);
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
  function_sum_ind_coal_exec(function, size, threads, blocks, shared_mem_usage, (void*)d_idata, (void*)d_odata, (void*)siteTable );

  //exit(1);



#if 0
  int smemSize = threads * sizeof(T2);

  QDPJitArgs cudaArgs;

  string typeT1,typeT2;

  getTypeString( typeT1 , *d_idata , cudaArgs );
  getTypeString( typeT2 , *d_odata , cudaArgs );

  ostringstream osId;
  osId << "reduce_convert_indirection " << typeT1 << " " << typeT2;
  string strId = osId.str();

#ifdef GPU_DEBUG_DEEP
  cout << "strId = " << strId << endl;
#endif

  QDP_debug("reduce_convert_indirection dev!");


  int aSize = cudaArgs.addInt(size); // numsitetable
  int aSiteTable = cudaArgs.addIntPtr( siteTable ); // soffsetDev
  int aInd = cudaArgs.addBool( indirection ); // indir
  int aIdata = cudaArgs.addPtr( (void*)d_idata );  // misc
  int aOdata = cudaArgs.addPtr( (void*)d_odata );  // dest
      
  std::ostringstream sprg;

  sprg << "    typedef " << typeT2 << " T2;" << endl;
  sprg << "    typedef " << typeT1 << " T1;" << endl;
  sprg << "    T1* g_idata = (T1*)(" << cudaArgs.getCode(aIdata) << ");" << endl;
  sprg << "    T2* g_odata = (T2*)(" << cudaArgs.getCode(aOdata) << ");" << endl;

  sprg << "    T2 *sdata = SharedMemory<T2>();" << endl;
  sprg << "    unsigned int tid = threadIdx.x;" << endl;
  sprg << "    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;" << endl;
  sprg << "    unsigned int j;" << endl;
  sprg << "    if (" << cudaArgs.getCode(aInd) << ")" << endl;
  sprg << "      j=" << cudaArgs.getCode(aSiteTable) << "[i];" << endl;
  sprg << "    else" << endl;
  sprg << "      j=i;" << endl;
  sprg << "    " << endl;
  sprg << "    if (i < " << cudaArgs.getCode(aSize) << ")" << endl;
  sprg << "      sdata[tid] = g_idata[j];" << endl;
  sprg << "    else" << endl;
  sprg << "      zero_rep(sdata[tid]);" << endl;
  sprg << "    " << endl;
  sprg << "    __syncthreads();" << endl;
  sprg << "" << endl;
  sprg << "    int next_pow2=1;" << endl;
  sprg << "    while( next_pow2 < blockDim.x ) {" << endl;
  sprg << "      next_pow2 <<= 1;" << endl;
  sprg << "    }" << endl;
  sprg << "" << endl;
  sprg << "    for(unsigned int s=next_pow2/2; s>0; s>>=1)" << endl;
  sprg << "    {" << endl;
  sprg << "        if (tid < s) " << endl;
  sprg << "        {" << endl;
  sprg << "	  if (tid + s < blockDim.x)" << endl;
  sprg << "            sdata[tid] += sdata[tid + s];" << endl;
  sprg << "        }" << endl;
  sprg << "        __syncthreads();" << endl;
  sprg << "    }" << endl;
  sprg << "" << endl;
  sprg << "    if (tid == 0) g_odata[blockIdx.x] = sdata[0];" << endl;

  string prg = sprg.str();

  static QDPJit::SharedLibEntry sharedLibEntry;
  if (!QDPJit::Instance().jitFixedGeom( strId , prg , cudaArgs.getDevPtr() , 
					  size , sharedLibEntry , threads, blocks , smemSize )) {
    QDP_error("reduce_convert_indirection() call to cuda jitter failed");
  }
#endif
  
  QDP_info("leaving reduce_convert<T1,T2>");
}









#if 1
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
      QDP_info("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );


      if (first) {
	if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat,subset) reduction buffer: 1st buffer no memory, exit");
	if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat,subset) reduction buffer: 2nd buffer no memory, exit");
      }



#if 1
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
#else
      if (numBlocks == 1) {
	if (first)
	  reduce_convert_indirection<T1,T2>(actsize, numThreads, numBlocks,  
					    (T1*)s1.getFdev() , (T2*)d.getFdev() , (int*)QDPCache::Instance().getDevicePtr(s.getId()) );
	else
	  reduce_convert_indirection<T2,T2>(actsize, numThreads, numBlocks, in_dev, (T2*)d.getFdev() );
      } else {
	if (first)
	  reduce_convert_indirection<T1,T2>(actsize, numThreads, numBlocks,  
					    (T1*)s1.getFdev(), out_dev , (int*)QDPCache::Instance().getDevicePtr(s.getId()) );
	else
	  reduce_convert_indirection<T2,T2>(actsize, numThreads, numBlocks, in_dev , out_dev );

      }
#endif

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
#endif

}

#endif
