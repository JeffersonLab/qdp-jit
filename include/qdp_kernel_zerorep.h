#ifndef QDP_KERNEL_ZEROREP_H
#define QDP_KERNEL_ZEROREP_H


namespace QDP {

  namespace ZeroRep {

    template<class T>
    void build_kernel( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest )
    {
      KernelBuilder k( "zero_rep" , __PRETTY_FUNCTION__ , plan );

      auto dest_jit = k.view( dest );

      zero_rep( dest_jit.elem( k.site() ) );

      k.finish( function );
    }


    template<class T>
    void exec( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , const Subset& s )
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

      LaunchArgs args( plan );
      args.add_iteration( th_count , layout.sitesOnNode() , s.getIdSiteTable(layout) );
      args.add_plan( plan , s );
      args.launch( function , th_count );

#ifdef QDP_DEEP_LOG
      QDP_get_global_logger().log(dest,s);
#endif
    }


    template<class T>
    void run( OLattice<T>& dest , const Subset& s )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_lattice( dest );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , dest );
	} );

      exec( function , plan , dest , s );
    }

  }

}
#endif
