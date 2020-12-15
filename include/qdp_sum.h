#ifndef SUM_H
#define SUM_H

namespace QDP {


  template <class T2>
  void qdp_jit_reduce(int size, int threads, int blocks, int shared_mem_usage, int in_id, int out_id )
  {
    static JitFunction function;

    if (function.empty())
      function = function_sum_build<T2>();

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
    static JitFunction function;

    if (function.empty())
      function = function_sum_convert_ind_build<T1,T2,input_layout>();

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
					    const multi1d<QDPCache::ArgKey>& table_ids)
  {
    static JitFunction function;

    assert( sizes.size() == numsubsets );
    assert( table_ids.size() == numsubsets );

    if (function.empty())
      function = function_summulti_convert_ind_build<T1,T2,input_layout>();

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
    assert( sizes.size() == numsubsets );
    static JitFunction function;

    if (function.empty())
      function = function_summulti_build<T>();

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
    static JitFunction function;

    if (function.empty())
      function = function_sum_convert_build<T1,T2,input_layout>();

    function_sum_convert_exec(function, size, threads, blocks, shared_mem_usage, 
			      in_id, out_id );
  }


  namespace {
    struct AndAssign {
      template<class J,class R>
      static void apply( J& j , const R& r )
      {
	j &= r;
      }
      template<class J>
      static void initNeutral( J& j )
      {
	typedef typename REGType<J>::Type_t R;
	j = R( true );
      }
    };
    struct IsFiniteAssign {
      template<class J,class R>
      static void apply( J& j , const R& r )
      {
	j = isfinite( r );
      }
    };
  }
  

  template <class T2, class ReductionOp >
  void qdp_jit_bool_reduction(int size, int threads, int blocks, int shared_mem_usage, int in_id, int out_id )
  {
    static JitFunction function;

    if (function.empty())
      function = function_bool_reduction_build<T2,ReductionOp>();

    function_bool_reduction_exec(function, size, threads, blocks, shared_mem_usage, in_id, out_id );
  }

  
  // T1 input
  // T2 output
  template < class T1 , class T2 , JitDeviceLayout input_layout , class ConvertOp, class ReductionOp >
  void qdp_jit_bool_reduction_convert(int size, 
				      int threads, 
				      int blocks, 
				      int shared_mem_usage,
				      int in_id, 
				      int out_id)
  {
    static JitFunction function;

    if (function.empty())
      function = function_bool_reduction_convert_build< T1 , T2 , input_layout , ConvertOp , ReductionOp >();

    function_bool_reduction_exec(function, size, threads, blocks, shared_mem_usage, in_id, out_id );
  }



  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSum>::Type_t
  sum(const OLattice<T1>& s1, const Subset& s)
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;
    
    int out_id,in_id;

    typename UnaryReturn<OLattice<T1>, FnSum>::Type_t  d;
    zero_rep(d);
    
    unsigned actsize=s.numSiteTable();
    bool first=true;
    bool allocated=false;
    while (actsize > 0) {

      unsigned numThreads = gpu_getMaxBlockX();
      while ((numThreads*sizeof(T2) > gpu_getMaxSMem()) || (numThreads > (unsigned)actsize)) {
	numThreads >>= 1;
      }
      unsigned numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > gpu_getMaxGridX()) {
	QDP_error_exit( "sum(Lat,subset) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)gpu_getMaxGridX());
      }

      int shared_mem_usage = numThreads*sizeof(T2);
      //QDP_info("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );

      if (first) {
	allocated=true;
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

    if (allocated)
      {
	QDP_get_global_cache().signoff( in_id );
	QDP_get_global_cache().signoff( out_id );
      }
    
    QDPInternal::globalSum(d);

    return d;
  }


  //
  // globalMax
  //
  template <class T>
  void globalMax_kernel(int size, int threads, int blocks, int in_id, int out_id)
  {
    int shared_mem_usage = threads * sizeof(T);

    static JitFunction function;

    if (function.empty())
      function = function_global_max_build<T>();

    function_global_max_exec(function, size, threads, blocks, shared_mem_usage, in_id, out_id );
  }



#if 1

  namespace {
    unsigned int nextPowerOf2(unsigned int n)  
    {
      unsigned count = 0;  
      
      if (n && !(n & (n - 1)))  
        return n;  
      
      while( n != 0)  
	{  
	  n >>= 1;  
	  count += 1;  
	}  
      
      return 1 << count;  
    }
  }
  
  //
  // sumMulti 
  //
  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t
  sumMulti( const OLattice<T1>& s1 , const Set& ss )
  {
    //QDPIO::cout << "using jit version of sumMulti\n";
    
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

    const int numsubsets = ss.numSubsets();

    typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t  dest( numsubsets );

    
    //QDPIO::cout << "number of subsets: " << numsubsets << "\n";

    multi1d<QDPCache::ArgKey> table_ids( numsubsets );
    multi1d<int>              sizes    ( numsubsets );

    // Zero-out the result (in case of empty subsets on the node)
    zero_rep( dest );

    //QDPIO::cout << "sizes = ";
    for (int i = 0 ; i < numsubsets ; ++i )
      {
	sizes[i]     = ss[i].numSiteTable();
	//QDPIO::cout << sizes[i] << " ";

	table_ids[i] = QDPCache::ArgKey( ss[i].getIdSiteTable() );
      }
    //QDPIO::cout << "\n";    
      
    bool first=true;
    bool allocated=false;
    int out_id,in_id;
    
    while( 1 ) {

      int maxsize = 0;
      for (int i = 0 ; i < numsubsets ; ++i )
	{
	  if ( sizes[i] > maxsize )
	    maxsize = sizes[i];
	}
      if (maxsize == 0)
	break;
      
      //QDPIO::cout << "maxsize: " << maxsize << "\n";

      int maxsizep2 = nextPowerOf2(maxsize);

      //QDPIO::cout << "maxsize power2 : " << maxsizep2 << "\n";
      
      unsigned numThreads = gpu_getMaxBlockX();
      while ((numThreads*sizeof(T2) > gpu_getMaxSMem()) || (numThreads > (unsigned)maxsizep2)) {
	numThreads >>= 1;
      }
      unsigned numBlocks=(int)ceil(float(maxsize)/numThreads);

      if (numBlocks > gpu_getMaxGridX()) {
	QDP_error_exit( "sumMulti(Lat,set) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)gpu_getMaxGridX());
      }

      int shared_mem_usage = numThreads*sizeof(T2);
      //QDP_info("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );

      if (first) {
	out_id = QDP_get_global_cache().add( numBlocks*sizeof(T2)*numsubsets , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	in_id  = QDP_get_global_cache().add( numBlocks*sizeof(T2)*numsubsets , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	allocated = true;
      }

      if (numBlocks == 1) {
	if (first) {
	  qdp_jit_summulti_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(maxsize, numThreads, numBlocks,
										 shared_mem_usage,
										 s1.getId(), dest.getId(),
										 numsubsets,
										 sizes,
										 table_ids);
	}
	else {
	  qdp_jit_summulti<T2>(maxsize, numThreads, numBlocks,
			       shared_mem_usage,
			       in_id, dest.getId(),
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
	multi1d<T1> tmp(numBlocks*numsubsets);
	std::vector<QDPCache::ArgKey> ids;
	ids.push_back(dest.getId());
	auto ptrs = QDP_get_global_cache().get_kernel_args( ids , false );
      
	CudaMemcpyD2H( (void*)tmp.slice() , ptrs[0] , numBlocks*numsubsets*sizeof(T1) );
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

    if (allocated)
      {
	QDP_get_global_cache().signoff( in_id );
	QDP_get_global_cache().signoff( out_id );
      }

    // This avoids an element-wise cuda memory copy
    dest.copyD2H();
    
    QDPInternal::globalSumArray(dest);

    return dest;
  }
#else
  template<class RHS, class T>
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
  sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
  {
    QDPIO::cout << "using non-jit version of sumMulti\n";
    typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t	 dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)	 
    static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
    prof.time -= getClockTime();
#endif

    // Initialize result with zero
    for(int k=0; k < ss.numSubsets(); ++k)
      zero_rep(dest[k]);

    // Loop over all sites and accumulate based on the coloring 
    const multi1d<int>& lat_color =	 ss.latticeColoring();
    const int nodeSites = Layout::sitesOnNode();

    for(int i=0; i < nodeSites; ++i) 
      {
	int j = lat_color[i];
	dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());
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

#endif



  template<class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
  globalMax(const OLattice<T>& s1)
  {
    int out_id, in_id;
	  
    const int nodeSites = Layout::sitesOnNode();

    typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

    int actsize=nodeSites;
    bool first=true;
    while (1) {

      int numThreads = gpu_getMaxBlockX();
      while ((numThreads*sizeof(T) > gpu_getMaxSMem()) || (numThreads > (unsigned)actsize)) {
	numThreads >>= 1;
      }
      int numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > gpu_getMaxGridX()) {
	QDP_error_exit( "globalMax(Lat) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)gpu_getMaxGridX());
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

    return d;
  }




  template<class T1>
  bool
  isfinite(const OLattice<T1>& s1)
  {
    typedef Boolean Ret_t;
    typedef typename Ret_t::SubType_t T2;
    
    Ret_t d;

    int out_id,in_id;
    unsigned actsize=Layout::sitesOnNode();
    bool first=true;
    while (1) {

      unsigned numThreads = gpu_getMaxBlockX();
      while ((numThreads*sizeof(T2) > gpu_getMaxSMem()) || (numThreads > (unsigned)actsize)) {
	numThreads >>= 1;
      }
      unsigned numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > gpu_getMaxGridX()) {
	QDP_error_exit( "isfinite(Lat) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)gpu_getMaxGridX());
      }

      int shared_mem_usage = numThreads*sizeof(T2);
      //QDP_info("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , shared_mem_usage );

      if (first) {
	out_id = QDP_get_global_cache().add( numBlocks*sizeof(T2) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	in_id  = QDP_get_global_cache().add( numBlocks*sizeof(T2) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
      }

      
      if (numBlocks == 1) {
	if (first) {
	  qdp_jit_bool_reduction_convert<T1,T2,JitDeviceLayout::Coalesced,IsFiniteAssign,AndAssign>(actsize, numThreads, numBlocks, shared_mem_usage, s1.getId(), d.getId() );
	}
	else {
	  qdp_jit_bool_reduction<T2,AndAssign>( actsize , numThreads , numBlocks, shared_mem_usage , in_id , d.getId() );
	}
      } else {
	if (first) {
	  qdp_jit_bool_reduction_convert<T1,T2,JitDeviceLayout::Coalesced,IsFiniteAssign,AndAssign>(actsize, numThreads, numBlocks, shared_mem_usage, s1.getId(), out_id );
	}
	else
	  qdp_jit_bool_reduction<T2,AndAssign>( actsize , numThreads , numBlocks , shared_mem_usage , in_id , out_id );

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

    bool ret_ = toBool(d);
    QDPInternal::globalAnd(ret_);

    return ret_;
  }


  template<class T1> bool isnormal(const OLattice<T1>& s1) { return isfinite(s1); }
  template<class T1> bool isnan(const OLattice<T1>& s1) { return !isfinite(s1); }
  template<class T1> bool isinf(const OLattice<T1>& s1) { return !isfinite(s1); }


    }

#endif
