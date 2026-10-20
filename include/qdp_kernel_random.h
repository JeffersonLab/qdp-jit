#ifndef QDP_KERNEL_RANDOM_H
#define QDP_KERNEL_RANDOM_H

namespace QDP {

  namespace Random {

    //! multipliers, the skewed and the lattice seed
    template<class T>
    ExprPlan describe( const OLattice<T>& dest , const Seed& seed_tmp ,
		       const LatticeSeed& latSeed , const LatticeSeed& skewedSeed , const RNG::RNG_Internals_t& rng )
    {
      ExprPlan plan;
      DescribeLeaves d( plan );
      forEach( dest , d , NullCombine() );
      d.add_scalar( seed_tmp.shape() , seed_tmp.get_ptr() , jit_sizeof(seed_tmp) ,
		    PlanEntry::ScalarMem::OwnHostUndef , true );
      forEach( rng.ran_mult_n , d , NullCombine() );
      forEach( rng.lattice_ran_mult , d , NullCombine() );
      forEach( skewedSeed , d , NullCombine() );
      forEach( latSeed , d , NullCombine() );
      return plan;
    }


    template<class T>
    void build_kernel( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , Seed& seed_tmp ,
		       LatticeSeed& latSeed , LatticeSeed& skewedSeed , const RNG::RNG_Internals_t& rng )
    {
      KernelBuilder k( "random" , __PRETTY_FUNCTION__ , plan );

      auto dest_jit             = k.view( dest );
      auto seed_tmp_jit         = k.view( seed_tmp );
      auto ran_mult_n_jit       = k.view( rng.ran_mult_n );
      auto lattice_ran_mult_jit = k.view( rng.lattice_ran_mult );
      auto skewedSeed_jit       = k.view( skewedSeed );
      auto latSeed_jit          = k.view( latSeed );

      llvm::Value* r_idx        = k.site();
      llvm::Value* r_idx_thread = k.thread();

      skewedSeed_jit.elem( r_idx ) = latSeed_jit.elem(  r_idx ) * lattice_ran_mult_jit.elem( r_idx );

      fill_random_jit( dest_jit.elem(r_idx) ,
		       latSeed_jit.elem( r_idx ) ,
		       skewedSeed_jit.elem( r_idx ) ,
		       ran_mult_n_jit.elem() );

      JitIf save( llvm_eq( r_idx_thread , llvm_create_value(0) ) );
      {
	seed_tmp_jit.elem() = latSeed_jit.elem( r_idx );      // seed_reg
      }
      save.end();

      k.finish( function );
    }


    template<class T>
    void exec( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , const Subset& s )
    {
#ifdef QDP_DEEP_LOG
      function.type_W = typeid(typename WordType<T>::Type_t).name();
      function.set_dest_id( dest.getId() );
      function.set_is_lat(true);
#endif

      const LatticeLayout layout = dest.layout();
      const int th_count = s.numSiteTable(layout);

      LaunchArgs args( plan );
      args.add_iteration( th_count , layout.sitesOnNode() , s.getIdSiteTable(layout) );
      args.add_plan( plan , s );
      args.launch( function , th_count );

      args.copy_written_to_host();

#ifdef QDP_DEEP_LOG
      QDP_get_global_logger().log(dest,s);
#endif
    }


    template<class T>
    void run( OLattice<T>& dest , const Subset& s )
    {
      const LatticeLayout layout = dest.layout();

      // No sites: no numbers drawn, the seed stays
      if (s.numSiteTable( layout ) < 1)
	return;

      static JitFunctionCache cache;

      LayoutScope scope( layout );

      RNG::RNG_Internals_t& rng = RNG::get_RNG_Internals( layout );

      Seed seed_tmp;
      LatticeSeed skewedSeed;
      LatticeSeed latSeed = rng.ran_seed;

      ExprPlan plan = describe( dest , seed_tmp , latSeed , skewedSeed , rng );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , dest , seed_tmp , latSeed , skewedSeed , rng );
	} );

      exec( function , plan , dest , s );

      rng.ran_seed = seed_tmp;
    }



    template<class T>
    ExprPlan describe( const OScalar<T>& dest , const Seed& seed , const Seed& skewedSeed , const Seed& ran_mult )
    {
      ExprPlan plan;
      DescribeLeaves d( plan );
      d.add_scalar( dest.shape() , dest.get_ptr() , jit_sizeof(dest) , PlanEntry::ScalarMem::Ring , true );
      d.add_scalar( seed.shape() , seed.get_ptr() , jit_sizeof(seed) , PlanEntry::ScalarMem::Ring , true );
      forEach( skewedSeed , d , NullCombine() );
      forEach( ran_mult , d , NullCombine() );
      return plan;
    }


    template<class T>
    void build_kernel( JitFunction& function , const ExprPlan& plan , OScalar<T>& dest , Seed& seed , Seed& skewedSeed , const Seed& ran_mult )
    {
      llvm_start_new_function( "random_sca" , __PRETTY_FUNCTION__ , qdp_jit_config_get_hostjit() );

      // No lattice leaves, hence no latsize param
      MGParamLeaf param_leaf( plan , -1 );

      const PlanEntry& e_dest = param_leaf.next( PlanEntry::Kind::Scalar );
      typedef typename JITType< OScalar<T> >::Type_t  OScalarJIT;

      OScalarJIT dest_jit( llvm_add_param< typename WordType<T>::Type_t * >() , e_dest.shape );
      auto seed_jit       = forEach( seed       , param_leaf , TreeCombine() );
      auto skewedSeed_jit = forEach( skewedSeed , param_leaf , TreeCombine() );
      auto ran_mult_jit   = forEach( ran_mult   , param_leaf , TreeCombine() );
      assert( param_leaf.all_entries_used() );

      llvm_create_function();

      fill_random_jit( dest_jit.elem() ,
		       seed_jit.elem() ,
		       skewedSeed_jit.elem() ,
		       ran_mult_jit.elem() );

      jit_get_function( function );

      assert( function.get_num_params() < 0 ||
	      function.get_num_params() == plan.num_leaf_params() );
    }


    template<class T>
    void exec( JitFunction& function , const ExprPlan& plan , OScalar<T>& dest )
    {
      LaunchArgs args( plan );
      args.add_plan( plan , all );
      args.launch( function , 1 );

      // dest, then seed back to the host
      args.copy_written_to_host();

#ifdef QDP_DEEP_LOG
      QDP_get_global_logger().log(dest);
#endif
    }


    template<class T>
    void run( OScalar<T>& dest )
    {
      static JitFunctionCache cache;

      RNG::RNG_Internals_t& rng = RNG::get_RNG_Internals();

      Seed seed = rng.ran_seed;
      Seed skewed_seed = rng.ran_seed * rng.ran_mult;

      ExprPlan plan = describe( dest , seed , skewed_seed , rng.ran_mult );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_kernel( f , plan , dest , seed , skewed_seed , rng.ran_mult );
	} );

      exec( function , plan , dest );

      rng.ran_seed = seed;
    }



    template<class T>
    void build_gaussian_kernel( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , OLattice<T>& r1 , OLattice<T>& r2 )
    {
      KernelBuilder k( "gaussian" , __PRETTY_FUNCTION__ , plan );

      auto dest_jit = k.view( dest );
      auto r1_jit   = k.view( r1 );
      auto r2_jit   = k.view( r2 );

      llvm::Value* r_idx = k.site();

      fill_gaussian_jit( dest_jit.elem( r_idx ) , r1_jit.elem( r_idx ) , r2_jit.elem( r_idx ) );

      k.finish( function );
    }


    template<class T>
    void exec_gaussian( JitFunction& function , const ExprPlan& plan , OLattice<T>& dest , const Subset& s )
    {
      const LatticeLayout layout = dest.layout();
      const int th_count = s.numSiteTable(layout);

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
    void run_gaussian( OLattice<T>& dest , const Subset& s )
    {
      const LatticeLayout layout = dest.layout();

      if (s.numSiteTable( layout ) < 1)
	return;

      LayoutScope scope( layout );

      OLattice<T> r1( dest.shape() );
      OLattice<T> r2( dest.shape() );

      run( r1 , s );
      run( r2 , s );

      static JitFunctionCache cache;

      ExprPlan plan = describe_leaves( dest , r1 , r2 );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_gaussian_kernel( f , plan , dest , r1 , r2 );
	} );

      exec_gaussian( function , plan , dest , s );
    }



    template<class T>
    void build_seedinit_kernel( JitFunction& function , const ExprPlan& plan , OLattice<T>& lat_ran_seed , OScalar<T>& ran_mult_n )
    {
      KernelBuilder k( "seedinit" , __PRETTY_FUNCTION__ , plan , false );

      auto lat_ran_seed_jit = k.view( lat_ran_seed );
      auto ran_mult_n_jit   = k.view( ran_mult_n );

      ParamRef p_sitesOnNode = k.add_param<int>();

      llvm::Value* r_sitesOnNode = llvm_derefParam( p_sitesOnNode );

      JitForLoop i( 1 , r_sitesOnNode );
      {
	lat_ran_seed_jit.elem( i.index() ) = lat_ran_seed_jit.elem( llvm_sub( i.index() , llvm_create_value(1) ) ) * ran_mult_n_jit.elem();
      }
      i.end();

      k.finish( function );
    }


    template<class T>
    void exec_seedinit( JitFunction& function , const ExprPlan& plan , OLattice<T>& lat_ran_seed )
    {
      const int sitesOnNode = lat_ran_seed.layout().sitesOnNode();

      LaunchArgs args( plan );
      args.add_iteration( 1 , sitesOnNode );
      args.add_plan( plan , all );
      args.add_int( sitesOnNode );
      args.launch( function , 1 );
    }


    template<class T>
    void run_seedinit( OLattice<T>& lat_ran_seed , OScalar<T>& ran_mult_n )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_leaves( lat_ran_seed , ran_mult_n );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_seedinit_kernel( f , plan , lat_ran_seed , ran_mult_n );
	} );

      exec_seedinit( function , plan , lat_ran_seed );
    }

  }

}

#endif
