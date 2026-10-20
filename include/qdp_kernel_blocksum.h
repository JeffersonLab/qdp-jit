#ifndef QDP_KERNEL_BLOCKSUM_H
#define QDP_KERNEL_BLOCKSUM_H

namespace QDP {

  namespace BlockSum {

    struct Geometry
    {
      int group_size;
      int threads_per_block;
    };

    Geometry geometry( int tile_size , int tiles , size_t value_size );


    template<class T2, class RHS, class T1>
    void build_kernel( JitFunction& function , const DynKey& key , const ExprPlan& dest_plan , const ExprPlan& rhs_plan ,
		       OLattice<T2>& dest , const QDPExpr<RHS,OLattice<T1> >& rhs , bool one_thread )
    {
      typedef typename WordType<T2>::Type_t WT;
      typedef typename JITType<T2>::Type_t T2JIT;

      const Shape shape = dest_plan.entries[0].shape;

      std::ostringstream expr;
      expr << std::string(__PRETTY_FUNCTION__) << "_key=" << key;
      llvm_start_new_function( "blocksum" , expr.str().c_str() );

      ParamRef p_tiles       = llvm_add_param<int>();
      ParamRef p_tile_size   = llvm_add_param<int>();
      ParamRef p_group_size  = llvm_add_param<int>();   // a power of 2
      ParamRef p_table_rhs   = llvm_add_param<int*>();
      ParamRef p_table_dest  = llvm_add_param<int*>();

      ParamRef p_latsize_rhs = llvm_add_param<int>();
      Reduce::ExprAtSite<RHS,T1> rhs_at( rhs_plan , rhs , p_latsize_rhs );

      ParamRef p_latsize_dest = llvm_add_param<int>();
      MGParamLeaf param_leaf_dest( dest_plan , p_latsize_dest );
      typedef typename LeafFunctor<OLattice<T2>, MGParamLeaf>::Type_t Dest_t;
      Dest_t dest_jit( forEach( dest , param_leaf_dest , TreeCombine() ) );
      assert( param_leaf_dest.all_entries_used() );

      llvm::Value* r_tiles      = llvm_derefParam( p_tiles );  // starts the function body
      llvm::Value* r_tile_size  = llvm_derefParam( p_tile_size );
      llvm::Value* r_group_size = one_thread ? llvm_create_value(1) : llvm_derefParam( p_group_size );

      llvm::Value* r_tidx   = llvm_call_special_tidx();
      llvm::Value* r_thread = llvm_add( llvm_mul( llvm_call_special_ctaidx() , llvm_call_special_ntidx() ) , r_tidx );
      llvm::Value* r_tile   = llvm_div( r_thread , r_group_size );
      llvm::Value* r_lane   = llvm_rem( r_thread , r_group_size );

      T2JIT sum = StackAllocJit<T2JIT>::alloc( shape );
      zero_rep( sum );

      JitIf ifTile( llvm_lt( r_tile , r_tiles ) );
      {
	llvm::Value* r_steps = llvm_div( llvm_sub( llvm_add( r_tile_size , r_group_size ) , llvm_create_value(1) ) , r_group_size );

	JitForLoop step( 0 , r_steps );
	{
	  llvm::Value* r_i = llvm_add( llvm_mul( step.index() , r_group_size ) , r_lane );

	  JitIf ifInTile( llvm_lt( r_i , r_tile_size ) );
	  {
	    llvm::Value* r_site = llvm_array_type_indirection<int>( p_table_rhs , llvm_add( llvm_mul( r_tile , r_tile_size ) , r_i ) );
	    sum += rhs_at( r_site );
	  }
	  ifInTile.end();
	}
	step.end();
      }
      ifTile.end();

      if ( one_thread )
      {
	JitIf ifTile2( llvm_lt( r_tile , r_tiles ) );
	{
	  dest_jit.elem( llvm_array_type_indirection<int>( p_table_dest , r_tile ) ) = sum;
	}
	ifTile2.end();
      }
      else
      {
	Reduce::SharedValues shared{ llvm_get_shared_ptr( llvm_get_type<WT>() , gpu_getMaxSMem() / sizeof(WT) ) ,
				     JitDataOrder::SiteFast , llvm_call_special_ntidx() };
	T2JIT value( shared.ptr , shared.order , shared.at( r_tidx ) , shape );
	value = sum;

	Reduce::emit_group_tree<Reduce::Sum>( value , shared , r_tidx , r_lane , r_group_size , shape );

	JitIf ifTile2( llvm_lt( r_tile , r_tiles ) );
	{
	  JitIf ifFirst( llvm_eq( r_lane , llvm_create_value(0) ) );
	  {
	    dest_jit.elem( llvm_array_type_indirection<int>( p_table_dest , r_tile ) ) = value;
	  }
	  ifFirst.end();
	}
	ifTile2.end();
      }

      function.set_explicit_geom();
      jit_get_function( function );

      assert( function.get_num_params() < 0 ||
	      function.get_num_params() == 6 + rhs_plan.num_leaf_params() + 1 + dest_plan.num_leaf_params() );
    }


    typedef std::function< JitFunction&( bool one_thread ) > Kernel;

    void launch( const Kernel& kernel , const ExprPlan& dest_plan , const ExprPlan& rhs_plan ,
		 LatticeLayout layout_dest , LatticeLayout layout_rhs , const multi1d<int>& blocking , size_t value_size );



    template<class RHS, class T1>
    OLattice< typename UnaryReturn<T1, FnSum >::Type_t >
    run( const QDPExpr<RHS, OLattice<T1> >& rhs , const multi1d<int>& blocking )
    {
      typedef typename UnaryReturn<T1, FnSum>::Type_t T2;

      const Shape shape = shape_of(rhs);
      const LatticeLayout layout_rhs = get_layout_from_rhs( rhs );

      LayoutScope coarse( layout_rhs.coarsen( blocking ) );
      OLattice<T2> dest(shape);
      coarse.close();

      static JitFunctionCache cache;

      ExprPlan dest_plan = describe_lattice( dest );
      ExprPlan rhs_plan  = describe_rhs( rhs , layout_rhs.index() );

      auto kernel = [&]( bool one_thread ) -> JitFunction& {
	DynKey key = combined_key( dest_plan , rhs_plan );
	key.add( one_thread ? 1 : 0 );

	JitFunction& function = cache.get( key , [&]() { return combined_structure( dest_plan , rhs_plan ); } , [&]( JitFunction& f ) {
	    build_kernel( f , key , dest_plan , rhs_plan , dest , rhs , one_thread );
	  } );

#ifdef QDP_DEEP_LOG
	function.type_W = typeid(typename WordType<T2>::Type_t).name();
	function.set_dest_id( dest.getId() );
	function.set_is_lat(true);
#endif
	return function;
      };

      const size_t value_size = WordSize<T2>::value(shape) * sizeof( typename WordType<T2>::Type_t );
      launch( kernel , dest_plan , rhs_plan , dest.layout() , layout_rhs , blocking , value_size );

#ifdef QDP_DEEP_LOG
      {
	LayoutScope scope( dest.layout() );
	QDP_get_global_logger().log(dest,all);
      }
#endif

      return dest;
    }

  } // namespace BlockSum



}

#endif
