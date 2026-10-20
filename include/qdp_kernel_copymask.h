#ifndef QDP_KERNEL_COPYMASK_H
#define QDP_KERNEL_COPYMASK_H


namespace QDP {

  namespace CopyMask {

    template<class T, class T1>
    void build_kernel( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
    {
      // one thread per site of the dest layout, no site table
      KernelBuilder k( "copymask" , __PRETTY_FUNCTION__ , plan , false );

      auto dest_jit = k.view( dest );
      auto src_jit  = k.view( src );
      auto mask_jit = k.view( mask );

      llvm::Value* r_idx = k.thread();

      JitIf ifCopy( mask_jit.elem( r_idx ).elem().elem().elem().elem() );
      {
	dest_jit.elem( r_idx ) = src_jit.elem( r_idx );
      }
      ifCopy.end();

      k.finish( function );
    }


    template<class T>
    void exec( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest )
    {
#ifdef QDP_DEEP_LOG
      function.type_W = typeid(typename WordType<T>::Type_t).name();
      function.set_dest_id( dest.getId() );
      function.set_is_lat(true);
#endif

      const int th_count = dest.layout().sitesOnNode();

      LaunchArgs args( plan );
      args.add_iteration( th_count , th_count );
      args.add_plan( plan , all );
      args.launch( function , th_count );

#ifdef QDP_DEEP_LOG
      QDP_get_global_logger().log(dest,all);
#endif
    }


    template<class T, class T1>
    void run( OLattice<T>& dest , const OLattice<T1>& mask , const OLattice<T>& src )
    {
      static JitFunctionCache cache;

      // operands in the kernel's param order
      ExprPlan plan = describe_leaves( dest , src , mask );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , dest , mask , src );
	} );

      exec( function , plan , dest );
    }

  }

}
#endif
