#ifndef QDP_KERNEL_EXTRACT_H
#define QDP_KERNEL_EXTRACT_H


namespace QDP {

  namespace Extract {

    template<class T, class T2>
    void build_kernel( JitFunction& function , const ExprPlan& plan , multi1d<OScalar<T> >& dest , const OLattice<T2>& src )
    {
      KernelBuilder k( "extract" , __PRETTY_FUNCTION__ , plan );

      ParamRef p_odata_ptr = k.add_param< typename WordType<T>::Type_t* >();

      OLatticeJIT< typename JITType< T >::Type_t > odata( p_odata_ptr , k.latsize() , plan.entries[0].shape , JitDataOrder::SiteSlow );

      auto src_jit = k.view( src );

      llvm::Value* r_idx = k.site();

      odata.elem( k.thread() ) = src_jit.elem( r_idx );

      k.finish( function );
    }


    template<class T, class T2>
    void exec( JitFunction& function , const ExprPlan& plan , multi1d<OScalar<T> >& dest , const OLattice<T2>& src , const Subset& s )
    {
      const LatticeLayout layout = src.layout();
      const int th_count = s.numSiteTable(layout);

      if (th_count < 1)
	return;

#ifdef QDP_DEEP_LOG
      function.type_W = typeid(typename WordType<T>::Type_t).name();
      function.set_is_lat(false);
#endif

      OScalar<T> tmp(src.shape());
      const size_t tmpsize = jit_sizeof(tmp);
      const size_t datasize = tmpsize * th_count;
      void* ptr = malloc( datasize );

      int d_id = QDP_get_global_cache().addOwnHostMemStatus( datasize , ptr , QDPCache::Status::undef );

      {
	LaunchArgs args( plan );
	args.add_iteration( th_count , layout.sitesOnNode() , s.getIdSiteTable(layout) );
	args.add_id( d_id );
	args.add_plan( plan , s );
	args.launch( function , th_count );
      }

      QDP_get_global_cache().assureOnHost(d_id);
      QDP_get_global_cache().signoff( d_id );

      for (int i = 0 ; i < th_count ; ++i )
	{
	  size_t idx = s.siteTable(layout)[i];
	  memcpy( dest[ idx ].get_ptr() , ((char*)ptr) + tmpsize * i , tmpsize );
	}

      free(ptr);
    }


    template<class T, class T2>
    void run( multi1d<OScalar<T> >& dest , const OLattice<T2>& src , const Subset& s )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_lattice( src );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , dest , src );
	} );

      exec( function , plan , dest , src , s );
    }



    // plan: describe_lattice( dest )
    template<class T>
    void build_insert_kernel( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest )
    {
      KernelBuilder k( "insert" , __PRETTY_FUNCTION__ , plan );

      ParamRef p_idata_ptr = k.add_param< typename WordType<T>::Type_t* >();

      OLatticeJIT< typename JITType< T >::Type_t > idata( p_idata_ptr , k.latsize() , plan.entries[0].shape , JitDataOrder::SiteSlow );

      auto dest_jit = k.view( dest );

      llvm::Value* r_idx = k.site();

      dest_jit.elem( r_idx ) = idata.elem( k.thread() );

      k.finish( function );
    }


    template<class T>
    void exec_insert( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , const multi1d<OScalar<T> >& src , const Subset& s )
    {
      const LatticeLayout layout = dest.layout();
      const int th_count = s.numSiteTable(layout);

      if (th_count < 1)
	return;

#ifdef QDP_DEEP_LOG
      function.type_W = typeid(typename WordType<T>::Type_t).name();
      function.set_dest_id( dest.getId() );
      function.set_is_lat(true);
#endif

      OScalar<T> tmp(dest.shape());
      const size_t tmpsize = jit_sizeof(tmp);
      const size_t datasize = tmpsize * th_count;
      void* ptr = malloc( datasize );

      for (int i = 0 ; i < th_count ; ++i )
	{
	  size_t idx = s.siteTable(layout)[i];
	  memcpy( ((char*)ptr) + tmpsize * i , src[ idx ].get_ptr() , tmpsize );
	}

      int d_id = QDP_get_global_cache().addOwnHostMem( datasize , ptr );

      {
	LaunchArgs args( plan );
	args.add_iteration( th_count , layout.sitesOnNode() , s.getIdSiteTable(layout) );
	args.add_id( d_id );
	args.add_plan( plan , s );
	args.launch( function , th_count );
      }

      QDP_get_global_cache().signoff( d_id );

      free(ptr);

#ifdef QDP_DEEP_LOG
      QDP_get_global_logger().log(dest,s);
#endif
    }


    template<class T>
    void run_insert( OLattice<T>& dest , const multi1d<OScalar<T> >& src , const Subset& s )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_lattice( dest );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_insert_kernel( f , plan , dest );
	} );

      exec_insert( function , plan , dest , src , s );
    }

  }

}
#endif
