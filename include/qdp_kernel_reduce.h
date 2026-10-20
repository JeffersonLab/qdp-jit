#ifndef QDP_KERNEL_REDUCE_H
#define QDP_KERNEL_REDUCE_H

#include <functional>

namespace QDP {

  namespace Reduce {

    struct Sum
    {
      static constexpr bool has_neutral = true;

      template<class T1, class J, class R>
      static void first( J& j , const R& r , Shape )
      {
	OpAssign()( j , r );
      }

      template<class J>
      static void neutral( J& j )
      {
	zero_rep( j );
      }

      template<class J, class R>
      static void combine( J& j , const R& r )
      {
	j += r;
      }
    };


    struct Max
    {
      static constexpr bool has_neutral = false;

      template<class T1, class J, class R>
      static void first( J& j , const R& r , Shape )
      {
	OpAssign()( j , r );
      }

      template<class J, class R>
      static void combine( J& j , const R& r )
      {
	j = where( j > r , j , r );
      }
    };


    struct AllFinite
    {
      static constexpr bool has_neutral = true;

      template<class T1, class J, class R>
      static void first( J& j , const R& r , Shape shape )
      {
	if constexpr ( requires { isfinite( r ); } )
	  {
	    j = isfinite( r );
	  }
	else
	  {
	    auto value = StackAllocJit< typename JITType<T1>::Type_t >::alloc( shape );
	    OpAssign()( value , r );
	    j = isfinite( value );
	  }
      }

      template<class J>
      static void neutral( J& j )
      {
	j = true;
      }

      template<class J, class R>
      static void combine( J& j , const R& r )
      {
	j &= r;
      }
    };



    class Segments
    {
    public:
      Segments( const Subset& s , LatticeLayout layout );

      Segments( const Set& ss , LatticeLayout layout );

      bool multi() const { return is_multi; }

      int count() const { return sizes.size(); }
      int max_size() const;

      int size( int i ) const { return sizes[i]; }
      const multi1d<int>& all_sizes() const { return sizes; }
      const multi1d<QDPCache::ArgKey>& table_ids() const { return tables; }

    private:
      bool is_multi;
      multi1d<int> sizes;
      multi1d<QDPCache::ArgKey> tables;
    };



    inline NestedIndex value_index( llvm::Value* i )
    {
      NestedIndex index;
      index.push_back( llvm_create_value( 1 ) , i );
      return index;
    }


    struct SharedValues
    {
      llvm::Value* ptr;
      JitDataOrder order = JitDataOrder::SiteSlow;
      llvm::Value* r_extent = nullptr;

      NestedIndex at( llvm::Value* i ) const
      {
	if ( order == JitDataOrder::SiteSlow )
	  return value_index( i );

	NestedIndex index;
	index.push_back( r_extent , i );
	return index;
      }
    };


    template<class Op, class TJIT>
    void emit_group_tree( TJIT& value , const SharedValues& shared , llvm::Value* r_tidx , llvm::Value* r_lane ,
			  llvm::Value* r_group_size , Shape shape )
    {
      llvm_bar_sync();

      JitForLoopPower loop( llvm_shr( r_group_size , llvm_create_value(1) ) );
      {
	JitIf inFirstHalf( llvm_lt( r_lane , loop.index() ) );
	{
	  TJIT other( shared.ptr , shared.order , shared.at( llvm_add( r_tidx , loop.index() ) ) , shape );
	  Op::combine( value , other );
	}
	inFirstHalf.end();

	llvm_bar_sync();
      }
      loop.end();
    }


    template<class RHS, class T1>
    class ExprAtSite
    {
      typedef typename ForEach<QDPExpr<RHS,OLattice<T1> >, MGParamLeaf, TreeCombine>::Type_t View_t;

      MGParamLeaf param_leaf;
      View_t view;

    public:
      ExprAtSite( const ExprPlan& plan , const QDPExpr<RHS,OLattice<T1> >& rhs , ParamRef p_latsize ):
	param_leaf( plan , p_latsize ),
	view( forEach( rhs , param_leaf , TreeCombine() ) )
      {
	assert( param_leaf.all_entries_used() );
      }

      auto operator()( llvm::Value* r_site )
      {
	MGViewLeaf vl( r_site , param_leaf );
	return forEach( view , vl , OpCombine() );
      }
    };


    template<class Op, class T2, class Size, class Load>
    void emit_block_reduce( ParamRef p_out , Shape shape , Size size , Load load , ParamRef p_out_offset = -1 )
    {
      typedef typename WordType<T2>::Type_t WT;
      typedef typename JITType<T2>::Type_t T2JIT;

      llvm::Value* r_out     = llvm_derefParam( p_out );
      llvm::Value* r_tidx    = llvm_call_special_tidx();
      llvm::Value* r_ntidx   = llvm_call_special_ntidx();
      llvm::Value* r_block   = llvm_call_special_ctaidx();
      llvm::Value* r_nblock  = llvm_call_special_nctaidx();
      llvm::Value* r_segment = llvm_call_special_ctaidy();

      llvm::Value* r_idx    = llvm_add( llvm_mul( r_block , r_ntidx ) , r_tidx );
      llvm::Value* r_stride = llvm_mul( r_nblock , r_ntidx );
      llvm::Value* r_size   = size( r_segment );

      T2JIT acc = StackAllocJit<T2JIT>::alloc( shape );

      JitIf inSegment( llvm_lt( r_idx , r_size ) );
      {
	load( acc , r_segment , r_idx );

	llvm::Value* r_steps = llvm_div( llvm_sub( llvm_add( r_size , r_stride ) , llvm_create_value(1) ) , r_stride );

	JitForLoop step( 1 , r_steps );
	{
	  llvm::Value* r_i = llvm_add( r_idx , llvm_mul( step.index() , r_stride ) );

	  JitIf ifInSegment( llvm_lt( r_i , r_size ) );
	  {
	    T2JIT next = StackAllocJit<T2JIT>::alloc( shape );
	    load( next , r_segment , r_i );
	    Op::combine( acc , next );
	  }
	  ifInSegment.end();
	}
	step.end();
      }
      inSegment.els();
      {
	if constexpr ( Op::has_neutral )
	  Op::neutral( acc );
	else
	  load( acc , r_segment , llvm_create_value( 0 ) );
      }
      inSegment.end();

      // one value per thread in shared memory
      llvm::Value* r_shared = llvm_get_shared_ptr( llvm_get_type<WT>() , gpu_getMaxSMem() / sizeof(WT) );
      T2JIT value( r_shared , JitDataOrder::SiteSlow , value_index( r_tidx ) , shape );
      value = acc;

      emit_group_tree<Op>( value , SharedValues{ r_shared } , r_tidx , r_tidx , r_ntidx , shape );

      JitIf ifStore( llvm_eq( r_tidx , llvm_create_value(0) ) );
      {
	llvm::Value* r_out_idx = llvm_add( llvm_mul( r_segment , r_nblock ) , r_block );
	if ( p_out_offset >= 0 )
	  r_out_idx = llvm_add( llvm_derefParam( p_out_offset ) , r_out_idx );
	T2JIT out( r_out , JitDataOrder::SiteSlow , value_index( r_out_idx ) , shape );
	out = value;
      }
      ifStore.end();
    }



    template<class Op, class T2, class RHS, class T1>
    void build_expr_kernel( JitFunction& function , const ExprPlan& plan , const QDPExpr<RHS,OLattice<T1> >& rhs , bool multi , Shape shape )
    {
#ifdef QDP_DEEP_LOG
      function.type_W = typeid(typename WordType<T2>::Type_t).name();
#endif
      llvm_start_new_function( multi ? "reduce_expr_multi" : "reduce_expr" , __PRETTY_FUNCTION__ );

      ParamRef p_size    = multi ? llvm_add_param< int* >()  : llvm_add_param< int >();
      ParamRef p_table   = multi ? llvm_add_param< int** >() : llvm_add_param< int* >();
      ParamRef p_latsize = llvm_add_param< int >();

      ExprAtSite<RHS,T1> rhs_at( plan , rhs , p_latsize );

      ParamRef p_out_offset = llvm_add_param< int >();
      ParamRef p_out        = llvm_add_param< typename WordType<T2>::Type_t* >();

      auto size = [&]( llvm::Value* r_segment ) {
	return multi ? llvm_array_type_indirection<int>( p_size , r_segment ) : llvm_derefParam( p_size );
      };

      auto load = [&]( typename JITType<T2>::Type_t& j , llvm::Value* r_segment , llvm::Value* r_idx ) {
	llvm::Value* r_table = multi ? llvm_array_type_indirection<int*>( p_table , r_segment ) : llvm_derefParam( p_table );
	llvm::Value* r_site  = llvm_array_type_indirection<int>( r_table , r_idx );

	Op::template first<T1>( j , rhs_at( r_site ) , shape );
      };

      emit_block_reduce<Op,T2>( p_out , shape , size , load , p_out_offset );

      function.set_explicit_geom();
      jit_get_function( function );

      assert( function.get_num_params() < 0 ||
	      function.get_num_params() == 3 + plan.num_leaf_params() + 2 );
    }


    template<class Op, class T2>
    void build_partials_kernel( JitFunction& function , Shape shape )
    {
      typedef typename WordType<T2>::Type_t WT;
      typedef typename JITType<T2>::Type_t T2JIT;

#ifdef QDP_DEEP_LOG
      function.type_W = typeid(WT).name();
#endif
      llvm_start_new_function( "reduce_partials" , __PRETTY_FUNCTION__ );

      ParamRef p_n   = llvm_add_param< int >();
      ParamRef p_in  = llvm_add_param< WT* >();
      ParamRef p_out = llvm_add_param< WT* >();

      auto size = [&]( llvm::Value* ) {
	return llvm_derefParam( p_n );
      };

      auto load = [&]( T2JIT& j , llvm::Value* r_segment , llvm::Value* r_idx ) {
	llvm::Value* r_in_idx = llvm_add( llvm_mul( r_segment , llvm_derefParam( p_n ) ) , r_idx );
	T2JIT in( llvm_derefParam( p_in ) , JitDataOrder::SiteSlow , value_index( r_in_idx ) , shape );
	j = in;
      };

      emit_block_reduce<Op,T2>( p_out , shape , size , load );

      function.set_explicit_geom();
      jit_get_function( function );
    }



    bool exec_expr( JitFunction& function , const ExprPlan& plan , const Segments& segments ,
		    int latsize , const kernel_geom_t& geom , size_t shared , int out_id , int out_offset );

    bool exec_partials( JitFunction& function , const kernel_geom_t& geom , size_t shared , int n , int in_id , int out_id );


    template<class Op, class T2, class RHS, class T1>
    bool launch_expr( const ExprPlan& plan , const QDPExpr<RHS,OLattice<T1> >& rhs , const Segments& segments ,
		      Shape shape , const kernel_geom_t& geom , size_t shared , int out_id , int out_offset )
    {
      static JitFunctionCache cache;

      DynKey key = plan.key();
      key.add( segments.multi() ? 1 : 0 );

      JitFunction& function = cache.get( key , [&]() { return plan.structure(); } , [&]( JitFunction& f ) {
	  build_expr_kernel<Op,T2>( f , plan , rhs , segments.multi() , shape );
	} );

      return exec_expr( function , plan , segments , get_layout_from_rhs( rhs ).sitesOnNode() , geom , shared , out_id , out_offset );
    }


    template<class Op, class T2>
    bool launch_partials( Shape shape , const kernel_geom_t& geom , size_t shared , int n , int in_id , int out_id )
    {
      static JitFunctionCache cache;

      DynKey key;
      key.add( shape );

      JitFunction& function = cache.get( key , []() { return std::vector<int>(); } , [&]( JitFunction& f ) {
	  build_partials_kernel<Op,T2>( f , shape );
	} );

      return exec_partials( function , geom , shared , n , in_id , out_id );
    }



    typedef std::function< bool( const kernel_geom_t& geom , size_t shared , int out_id , int k , int out_offset ) > FirstPass;
    typedef std::function< bool( const kernel_geom_t& geom , size_t shared , int n , int in_id , int out_id ) > LaterPass;

    void run_passes( int count , int batch , int max_size , size_t value_size , size_t bytes_first , void* result ,
		     const FirstPass& first , const LaterPass& later );


    size_t bytes_per_site( const ExprPlan& plan );


    template<class Op, class T2, class RHS, class T1>
    void run( const QDPExpr<RHS,OLattice<T1> >& rhs , const Segments& segments , Shape shape , size_t value_size , void* result )
    {
      ExprPlan plan = describe_rhs( rhs , get_layout_from_rhs( rhs ).index() );

      run_passes( segments.count() , 1 , segments.max_size() , value_size , bytes_per_site( plan ) , result ,
		  [&]( const kernel_geom_t& geom , size_t shared , int out_id , int , int out_offset ) {
		    return launch_expr<Op,T2>( plan , rhs , segments , shape , geom , shared , out_id , out_offset );
		  },
		  [&]( const kernel_geom_t& geom , size_t shared , int n , int in_id , int out_id ) {
		    return launch_partials<Op,T2>( shape , geom , shared , n , in_id , out_id );
		  } );
    }



    template<class Op, class T2, class Make>
    void run_batch( int n , const Make& make , const Subset& s , Shape shape , size_t value_size , void* result )
    {
      const auto rhs0 = make(0);
      const LatticeLayout layout = get_layout_from_rhs( rhs0 );
      const Segments segments( s , layout );

      run_passes( 1 , n , segments.max_size() , value_size , bytes_per_site( describe_rhs( rhs0 , layout.index() ) ) , result ,
		  [&]( const kernel_geom_t& geom , size_t shared , int out_id , int k , int out_offset ) {
		    const auto rhs = make(k);
		    ExprPlan plan = describe_rhs( rhs , layout.index() );
		    return launch_expr<Op,T2>( plan , rhs , segments , shape , geom , shared , out_id , out_offset );
		  },
		  [&]( const kernel_geom_t& geom , size_t shared , int n , int in_id , int out_id ) {
		    return launch_partials<Op,T2>( shape , geom , shared , n , in_id , out_id );
		  } );
    }


    template<class E> struct ExprOfLattice;
    template<class RHS, class T1> struct ExprOfLattice< QDPExpr<RHS,OLattice<T1> > > { typedef T1 Type_t; };

    template<class Make>
    auto sum_elements( int n , const Make& make , const Subset& s )
    {
      typedef typename ExprOfLattice< decltype( make(0) ) >::Type_t T1;
      typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t Ret_t;

      const Shape shape = shape_of( make(0) );

      Ret_t d(shape);
      zero_rep(d);   // the sum over no sites

      run_batch<Sum, typename Ret_t::SubType_t>( n , make , s , shape , jit_sizeof(d) , d.get_ptr() );

      QDPInternal::globalSum(d);
      return d;
    }

  }

} // namespace QDP

#endif
