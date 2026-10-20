#ifndef QDP_KERNEL_SITE_H
#define QDP_KERNEL_SITE_H


namespace QDP {

  namespace Site {

    template<class T, class T1>
    ExprPlan describe_peek( const OLattice<T>& l , const OScalar<T1>& dest )
    {
      ExprPlan plan;
      DescribeLeaves d( plan );
      d.add_scalar( dest.shape() , dest.get_ptr() ,
		    WordSize<T1>::value(dest.shape()) * sizeof( typename WordType<T1>::Type_t ) ,
		    PlanEntry::ScalarMem::OwnHost , true );
      forEach( l , d , NullCombine() );
      return plan;
    }


    template<class T, class T1>
    void build_peek_kernel( JitFunction& function , const ExprPlan& plan , const OLattice<T>& l , const OScalar<T1>& dest )
    {
      llvm_start_new_function( "eval_peekSite" , __PRETTY_FUNCTION__ );

      ParamRef p_siteindex  = llvm_add_param<int>();
      ParamRef p_latsize    = llvm_add_param<int>();

      MGParamLeaf param_leaf( plan , p_latsize );

      // dest: pointer param (it is written)
      const PlanEntry& e_dest = param_leaf.next( PlanEntry::Kind::Scalar );
      typedef typename JITType< OScalar<T1> >::Type_t  DestJIT_t;
      DestJIT_t dest_jit( llvm_add_param< typename WordType<T1>::Type_t * >() , e_dest.shape );

      auto l_jit = forEach( l , param_leaf , TreeCombine() );
      assert( param_leaf.all_entries_used() );

      llvm::Value* r_siteindex = llvm_derefParam( p_siteindex );

      dest_jit.elem() = l_jit.elem( r_siteindex );

      jit_get_function( function );

      assert( function.get_num_params() < 0 ||
	      function.get_num_params() == 2 + plan.num_leaf_params() );
    }


    template<class T>
    void exec_peek( JitFunction& function , const ExprPlan& plan , const OLattice<T>& l , int index )
    {
      LaunchArgs args( plan );
      args.add_int( index );
      args.add_int( l.layout().sitesOnNode() );
      args.add_plan( plan , all );   // dest (own cache entry), l
      args.launch( function , 1 );

      // copy the result to the host; the dest entry is signed off after that
      args.copy_written_to_host();
    }


    template<class T, class T1>
    void run_peek( const OLattice<T>& l , OScalar<T1>& dest , const multi1d<int>& coord )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_peek( l , dest );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_peek_kernel( f , plan , l , dest );
	} );

      const LatticeLayout layout = l.layout();

      if ( layout.nodeNumber() == layout.nodeNumber(coord) )
	exec_peek( function , plan , l , layout.linearSiteIndex(coord) );
      else
	dest.zero_me();
    }



    template<class T, class T1>
    void build_poke_kernel( JitFunction& function , const ExprPlan& plan , const OLattice<T>& l , const OScalar<T1>& r )
    {
      llvm_start_new_function( "eval_pokeSite" , __PRETTY_FUNCTION__ );

      ParamRef p_siteindex  = llvm_add_param<int>();
      ParamRef p_latsize    = llvm_add_param<int>();

      MGParamLeaf param_leaf( plan , p_latsize );

      auto l_jit = forEach( l , param_leaf , TreeCombine() );

      auto op_jit = AddOpParam<OpAssign,MGParamLeaf>::apply( OpAssign() , param_leaf );

      typedef typename LeafFunctor<OScalar<T1>, MGParamLeaf>::Type_t  RJIT_t;
      RJIT_t r_jit( forEach( r , param_leaf , TreeCombine() ) );
      assert( param_leaf.all_entries_used() );

      llvm::Value* r_siteindex = llvm_derefParam( p_siteindex );
      llvm::Value* r_zero      = llvm_create_value(0);

      MGViewLeaf vl( r_zero , param_leaf );
      op_jit( l_jit.elem( r_siteindex ),
	      LeafFunctor< RJIT_t , MGViewLeaf >::apply( r_jit , vl ) );

      jit_get_function( function );

      assert( function.get_num_params() < 0 ||
	      function.get_num_params() == 2 + plan.num_leaf_params() );
    }


    template<class T>
    void exec_poke( JitFunction& function , const ExprPlan& plan , const OLattice<T>& l , int index )
    {
      LaunchArgs args( plan );
      args.add_int( index );
      args.add_int( l.layout().sitesOnNode() );
      args.add_plan( plan , all );
      args.launch( function , 1 );
    }


    template<class T, class T1>
    void run_poke( OLattice<T>& l , const OScalar<T1>& r , const multi1d<int>& coord )
    {
      static JitFunctionCache cache;

      ExprPlan plan = describe_leaves( l , r );

      JitFunction& function = cache.get( plan , [&]( JitFunction& f ) {
	  build_poke_kernel( f , plan , l , r );
	} );

      const LatticeLayout layout = l.layout();

      if ( layout.nodeNumber() == layout.nodeNumber(coord) )
	exec_poke( function , plan , l , layout.linearSiteIndex(coord) );
    }

  }

}
#endif
