#ifndef SUM_H
#define SUM_H

namespace QDP {

#if defined (QDP_BACKEND_CUDA) || defined (QDP_BACKEND_ROCM)
  template <class T2>
  void qdp_jit_reduce(int size, int threads, int blocks, int shared_mem_usage, int in_id, int out_id )
  {
    static JitFunction function;

    if (function.empty())
      function_sum_build<T2>(function);

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
       function_sum_convert_ind_build<T1,T2,input_layout>(function);

    function_sum_convert_ind_exec(function, size, threads, blocks, shared_mem_usage, 
				  in_id, out_id, siteTableId );
  }


  // T1 input
  // T2 output
  template < JitDeviceLayout input_layout, class T1 , class RHS >
  void qdp_jit_reduce_convert_indirection_expr(int size, 
					       int threads, 
					       int blocks, 
					       int shared_mem_usage,
					       const QDPExpr<RHS,OLattice<T1> >& rhs, 
					       int out_id, 
					       int siteTableId)
  {
    static JitFunction function;

    if (function.empty())
      function_sum_convert_ind_expr_build<input_layout>(function,rhs);

    function_sum_convert_ind_expr_exec(function, size, threads, blocks, shared_mem_usage, 
				       rhs, out_id, siteTableId );
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
      function_summulti_convert_ind_build<T1,T2,input_layout>(function);

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
      function_summulti_build<T>(function);

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
      function_sum_convert_build<T1,T2,input_layout>(function);

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
      function_bool_reduction_build<T2,ReductionOp>(function);

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
      function_bool_reduction_convert_build< T1 , T2 , input_layout , ConvertOp , ReductionOp >(function);

    function_bool_reduction_exec(function, size, threads, blocks, shared_mem_usage, in_id, out_id );
  }
#endif


  
#if defined (QDP_BACKEND_CUDA) || (QDP_BACKEND_ROCM)
  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSum>::Type_t
  sum(const OLattice<T1>& s1, const Subset& s)
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;
    
    int out_id,in_id;

    typename UnaryReturn<OLattice<T1>, FnSum>::Type_t  d;
    zero_rep(d);

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
    prof.start_time();
#endif

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t) , d.getF() , nullptr );
    
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
	  qdp_jit_reduce_convert_indirection<T1,T2,JitDeviceLayout::Coalesced>(actsize, numThreads, numBlocks, shared_mem_usage, s1.getId(), d_id , s.getIdSiteTable());
	}
	else {
	  qdp_jit_reduce<T2>( actsize , numThreads , numBlocks, shared_mem_usage , in_id , d_id );
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

    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    // Global sum
    QDPInternal::globalSum(d);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

#ifdef QDP_DEEP_LOG
    gpu_deep_logger( d.getF() , typeid(typename WordType<T2>::Type_t).name() , sizeof(T2) , __PRETTY_FUNCTION__ , false );
#endif

    return d;
  }
#elif defined (QDP_BACKEND_AVX)
  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSum>::Type_t
  sum(const OLattice<T1>& s1, const Subset& s)
  {
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

    typename UnaryReturn<OLattice<T1>, FnSum>::Type_t sum;

    zero_rep(sum);

    //#pragma omp parallel for reduction(+: sum)
    for (auto i = 0; i < s.numSiteTable(); i++)
      {
	int j = s.siteTable()[i];

	// The ::elem access is required, otherwise it calls
	// eval(OScalar,ExprOScalar) which then invalidates the logger
	//
	sum.elem() += s1.peekLinearSite(j).elem();
      }

    // Global sum
    QDPInternal::globalSum(sum);

#ifdef QDP_DEEP_LOG
    gpu_deep_logger( sum.getF() , typeid(typename WordType<T2>::Type_t).name() , sizeof(T2) , __PRETTY_FUNCTION__ , false );
#endif

    return sum;
  }
#elif defined (QDP_BACKEND_L0)
#warning "no sum"
#else
#error "No backend specified"
#endif
  

  template<class T1, class RHS>
  typename UnaryReturn< OLattice<T1> , FnSum>::Type_t
  sum(const QDPExpr<RHS,OLattice<T1> >& rhs)
  {
    return sum(rhs,all);
  }

  
  template<class T1, class RHS>
  typename UnaryReturn< OLattice<T1> , FnSum>::Type_t
  sum(const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
  {
    OLattice<T1> tmp = rhs;
    return sum(tmp,s);
  }

#if 0
#if defined (QDP_BACKEND_CUDA) || (QDP_BACKEND_ROCM)
  template<class WT>
  typename UnaryReturn< OLattice<PScalar<PScalar<RScalar<Word<double> > > > > , FnSum>::Type_t
  sum(const QDPExpr<UnaryNode<FnLocalNorm2, Reference<QDPType<PSpinVector<PColorVector<RComplex<Word<WT> >, 3>, 4>, OLattice<PSpinVector<PColorVector<RComplex<Word<WT> >, 3>, 4> > > > >,OLattice<PScalar<PScalar<RScalar<Word<double> > > > > >& rhs, const Subset& s)
  {
    typedef typename UnaryReturn< OLattice< PScalar<PScalar<RScalar<Word<double> > > > > , FnSum>::Type_t::SubType_t T2;
    
    int out_id,in_id;

    typename UnaryReturn< OLattice< PScalar<PScalar<RScalar<Word<double> > > > >, FnSum>::Type_t  d;
    zero_rep(d);

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(d, OpAssign(), FnSum(), rhs);
    prof.start_time();
#endif

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(typename UnaryReturn< OLattice< PScalar<PScalar<RScalar<Word<double> > > > > , FnSum>::Type_t::SubType_t) , d.getF() , nullptr );
    
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

      
      if (numBlocks == 1)
	{
	  if (first)
	    {
	      qdp_jit_reduce_convert_indirection_expr<JitDeviceLayout::Coalesced>(actsize, numThreads, numBlocks, shared_mem_usage, rhs , d_id , s.getIdSiteTable());
	    }
	  else
	    {
	      qdp_jit_reduce<T2>( actsize , numThreads , numBlocks, shared_mem_usage , in_id , d_id );
	    }
	}
      else
	{
	  if (first)
	    {
	      qdp_jit_reduce_convert_indirection_expr<JitDeviceLayout::Coalesced>(actsize, numThreads, numBlocks, shared_mem_usage, rhs , out_id, s.getIdSiteTable());
	    }
	  else
	    {
	      qdp_jit_reduce<T2>( actsize , numThreads , numBlocks , shared_mem_usage , in_id , out_id );
	    }

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

    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    // Global sum
    QDPInternal::globalSum(d);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );


#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

#ifdef QDP_DEEP_LOG
    gpu_deep_logger( d.getF() , typeid(typename WordType<T2>::Type_t).name() , sizeof(T2) , __PRETTY_FUNCTION__ , false );
#endif

    return d;
  }
#endif
#endif
  

  //
  // globalMax
  //
  template <class T>
  void globalMax_kernel(int size, int threads, int blocks, int in_id, int out_id)
  {
    int shared_mem_usage = threads * sizeof(T);

    static JitFunction function;

    if (function.empty())
      function_global_max_build<T>(function);

    function_global_max_exec(function, size, threads, blocks, shared_mem_usage, in_id, out_id );
  }



#if defined (QDP_BACKEND_CUDA) || (QDP_BACKEND_ROCM)
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

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(dest, OpAssign(), FnSumMulti(), s1);
    prof.start_time();
#endif

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(T2) * numsubsets , dest.slice() , nullptr );
    
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
										 s1.getId(), d_id,
										 numsubsets,
										 sizes,
										 table_ids);
	}
	else {
	  qdp_jit_summulti<T2>(maxsize, numThreads, numBlocks,
			       shared_mem_usage,
			       in_id, d_id,
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

      
      first =false;

      if (numBlocks==1)
	break;

      for (int i = 0 ; i < numsubsets ; ++i )
	{
	  sizes[i] = numBlocks;
	}

      int tmp = in_id;
      in_id = out_id;
      out_id = tmp;
    }

    if (allocated)
      {
	QDP_get_global_cache().signoff( in_id );
	QDP_get_global_cache().signoff( out_id );
      }


    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    // Global sum
    QDPInternal::globalSumArray(dest);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

#ifdef QDP_DEEP_LOG
    gpu_deep_logger( dest.slice() , typeid(typename WordType<T2>::Type_t).name() , sizeof(T2)*numsubsets , __PRETTY_FUNCTION__ , false );
#endif
    
    return dest;
  }
#elif defined (QDP_BACKEND_AVX)
  template<class T1>
  typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t
  sumMulti( const OLattice<T1>& s1 , const Set& ss )
  {
    typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t dest(ss.numSubsets());

    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

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
	dest[j].elem() += s1.peekLinearSite(i).elem();
      }

    // Do a global sum on the result
    QDPInternal::globalSumArray(dest);

#if defined(QDP_USE_PROFILING)	 
    prof.time += getClockTime();
    prof.count++;
    prof.print();
#endif

#ifdef QDP_DEEP_LOG
    gpu_deep_logger( dest.slice() , typeid(typename WordType<T2>::Type_t).name() , sizeof(T2) * ss.numSubsets() , __PRETTY_FUNCTION__ , false );
#endif

    return dest;
  }
#elif defined (QDP_BACKEND_L0)
#warning "no sumMulti"
#else
#error "No backend specified"
#endif


#if defined (QDP_BACKEND_CUDA) || (QDP_BACKEND_ROCM)
  template<class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
  globalMax(const OLattice<T>& s1)
  {
    int out_id, in_id;
	  
    const int nodeSites = Layout::sitesOnNode();

    typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
    prof.start_time();
#endif

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t) , d.getF() , nullptr );
    
    int actsize=nodeSites;
    bool first=true;
    while (1) {

      int numThreads = gpu_getMaxBlockX();
      while ((numThreads*sizeof(T) > gpu_getMaxSMem()) || (numThreads > (unsigned)actsize)) {
	numThreads >>= 1;
      }
      int numBlocks=(int)ceil(float(actsize)/numThreads);
      
      //QDP_info("max(Lat): using %d threads per block, %d blocks" , numThreads , numBlocks );

      if (numBlocks > gpu_getMaxGridX()) {
	QDP_error_exit( "globalMax(Lat) numBlocks(%d) > maxGridX(%d)",numBlocks,(int)gpu_getMaxGridX());
      }

      if (first) {
	out_id = QDP_get_global_cache().add( numBlocks*sizeof(T) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
	in_id  = QDP_get_global_cache().add( numBlocks*sizeof(T) , QDPCache::Flags::Empty , QDPCache::Status::undef , NULL , NULL , NULL );
      }


      if (numBlocks == 1) {
	if (first)
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, s1.getId() , d_id );
	else
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, in_id, d_id );
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

    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    // Global max
    QDPInternal::globalMax(d);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );

#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

#ifdef QDP_DEEP_LOG
    gpu_deep_logger( d.getF() , typeid(typename WordType<T>::Type_t).name() , sizeof(T) , __PRETTY_FUNCTION__ , false );
#endif
    
    return d;
  }
#elif defined (QDP_BACKEND_AVX)
  template<class T>
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
  globalMax(const OLattice<T>& s1)
  {
    typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t	d;
	
#if defined(QDP_USE_PROFILING)	 
    static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
    prof.time -= getClockTime();
#endif

    // Loop always entered so unroll
    d.elem() = s1.elem(0);

    const int vvol = Layout::sitesOnNode();
    for(int i=1; i < vvol; ++i) 
      {
	auto dd = s1.peekLinearSite(i).elem();
	
	if (toBool(dd > d.elem()))
	  d.elem() = dd;
      }

    // Do a global max on the result
    QDPInternal::globalMax(d); 

#if defined(QDP_USE_PROFILING)	 
    prof.time += getClockTime();
    prof.count++;
    prof.print();
#endif

#ifdef QDP_DEEP_LOG
    gpu_deep_logger( d.getF() , typeid(typename WordType<T>::Type_t).name() , sizeof(T) , __PRETTY_FUNCTION__ , false );
#endif

    return d;
  }
#elif defined (QDP_BACKEND_L0)
#warning "no globalMax"
#else
#error "no backend specified"
#endif


#if defined (QDP_BACKEND_CUDA) || (QDP_BACKEND_ROCM)
  template<class T1>
  bool
  isfinite(const OLattice<T1>& s1)
  {
    typedef Boolean::SubType_t T2;
    
    Boolean d;

#if defined(QDP_USE_PROFILING)
    static QDPProfile_t prof(d, OpAssign(), FnIsFinite(), s1);
    prof.start_time();
#endif

    // Register the destination object with the memory cache
    int d_id = QDP_get_global_cache().registrateOwnHostMem( sizeof(Boolean) , d.getF() , nullptr );

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
	  qdp_jit_bool_reduction_convert<T1,T2,JitDeviceLayout::Coalesced,IsFiniteAssign,AndAssign>(actsize, numThreads, numBlocks, shared_mem_usage, s1.getId(), d_id );
	}
	else {
	  qdp_jit_bool_reduction<T2,AndAssign>( actsize , numThreads , numBlocks, shared_mem_usage , in_id , d_id );
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

    // Copy result to host
    QDP_get_global_cache().assureOnHost(d_id);

    bool ret_ = toBool(d);
    QDPInternal::globalAnd(ret_);

    // Sign off result
    QDP_get_global_cache().signoff( d_id );

    
#if defined(QDP_USE_PROFILING)
    prof.end_time();
#endif

    return ret_;
  }
#elif defined (QDP_BACKEND_AVX)
  template<class T1>
  bool
  isfinite(const OLattice<T1>& s1)
  {
    bool d = true;

#if defined(QDP_USE_PROFILING)   
    static QDPProfile_t prof(&d, OpAssign(), FnIsFinite(), s1);
    prof.time -= getClockTime();
#endif

    const int nodeSites = Layout::sitesOnNode();
    for(int i=0; i < nodeSites; ++i) 
      {
	auto obj = s1.peekLinearSite(i);

	typedef typename WordType<T1>::Type_t WT;
	
	WT *ptr = (WT*)obj.getF();

	for (int w = 0 ; w < sizeof(T1)/sizeof(WT) ; ++w )
	  {
	    d &= std::isfinite(*ptr);
	    ptr++;
	  }
      }

    QDPInternal::globalAnd(d);

#if defined(QDP_USE_PROFILING)   
    prof.time += getClockTime();
    prof.count++;
    prof.print();
#endif

    return d;
  }
#elif defined (QDP_BACKEND_L0)
#warning "no isfinite"
#else
#error "no backend specified"
#endif

  

  template<class T1> bool isnormal(const OLattice<T1>& s1) { return isfinite(s1); }
  template<class T1> bool isnan(const OLattice<T1>& s1) { return !isfinite(s1); }
  template<class T1> bool isinf(const OLattice<T1>& s1) { return !isfinite(s1); }


} // QDP

#endif
