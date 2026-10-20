#ifndef QDP_JIT_PLAN_LEAVES_H
#define QDP_JIT_PLAN_LEAVES_H

namespace QDP {

  //
  //
  struct DescribeLeaves
  {
    ExprPlan& plan;
    DescribeLeaves( ExprPlan& p ): plan(p) {}

    void add_lattice( const Shape& shape , JitDataFormat data_format , int layout_ref , int id )
    {
      PlanEntry e;
      e.kind        = PlanEntry::Kind::Lattice;
      e.shape          = shape;
      e.data_format = data_format;
      e.layout_slot = plan.slot_of_layout( layout_ref );
      e.layout_ref  = layout_ref;
      e.cache_id    = id;
      plan.entries.push_back( e );
    }

    void add_scalar( const Shape& shape , const void* ptr , size_t bytes ,
		     PlanEntry::ScalarMem mem = PlanEntry::ScalarMem::Ring , bool written = false )
    {
      PlanEntry e;
      e.kind     = PlanEntry::Kind::Scalar;
      e.shape       = shape;
      e.host_ptr = ptr;
      e.bytes    = bytes;
      e.mem      = mem;
      e.written  = written;
      plan.entries.push_back( e );
    }

    template<class W>
    void add_literal( const Shape& shape , W value )
    {
      PlanEntry e;
      e.kind    = PlanEntry::Kind::Literal;
      e.shape      = shape;
      e.literal = value;
      plan.entries.push_back( e );
    }

    void add_gamma( int g )
    {
      PlanEntry e;
      e.kind  = PlanEntry::Kind::Gamma;
      e.gamma = g;
      plan.entries.push_back( e );
    }

    void add_shift( const FnMap& fnmap , bool offnode )
    {
      PlanEntry e;
      e.kind    = PlanEntry::Kind::Shift;
      e.offnode = offnode ? 1 : 0;
      e.fnmap   = const_cast<FnMap*>( &fnmap );
      plan.entries.push_back( e );
    }

    void add_op_param( int value )
    {
      PlanEntry e;
      e.kind     = PlanEntry::Kind::OpParam;
      e.op_param = value;
      plan.entries.push_back( e );
    }
  };


  template<class T>
  struct LeafFunctor<OLattice<T>, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const OLattice<T>& a, DescribeLeaves& d)
    {
      d.add_lattice( a.shape() , a.data_format() , a.layout().index() , a.getId() );
      return 0;
    }
  };

  template<class T>
  struct LeafFunctor<QDPType<T,OLattice<T> >, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const QDPType<T,OLattice<T> >& s, DescribeLeaves& d)
    {
      const OLattice<T>* a = static_cast<const OLattice<T>*>(&s);
      d.add_lattice( a->shape() , a->data_format() , a->layout().index() , a->getId() );
      return 0;
    }
  };

  template<class T>
  struct LeafFunctor<OScalar<T>, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const OScalar<T>& a, DescribeLeaves& d)
    {
      d.add_scalar( a.shape() , a.get_ptr() , jit_sizeof( a ) );
      return 0;
    }
  };

  template<class T>
  struct LeafFunctor<QDPType<T,OScalar<T> >, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const QDPType<T,OScalar<T> >& s, DescribeLeaves& d)
    {
      size_t size = WordSize<T>::value(s.shape()) * sizeof( typename WordType<T>::Type_t );
      d.add_scalar( s.shape() , static_cast<const OScalar<T>*>(&s)->get_ptr() , size );
      return 0;
    }
  };

  template<class T>
  struct LeafFunctor<QDPType<PScalar< PScalar < RScalar< Word< T > > > >,OScalar<PScalar< PScalar < RScalar< Word< T > > > > > >, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const QDPType<PScalar< PScalar < RScalar< Word< T > > > > , OScalar<PScalar< PScalar < RScalar< Word< T > > > > > >& s, DescribeLeaves& d)
    {
      const OScalar<PScalar< PScalar < RScalar< Word< T > > > > >* a =
	static_cast<const OScalar<PScalar< PScalar < RScalar< Word< T > > > > >*>(&s);
      d.add_literal< T >( a->shape() , a->get_word_value() );
      return 0;
    }
  };

  template<class T>
  struct LeafFunctor<OScalar< PScalar< PScalar < RScalar< Word< T > > > > >, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const OScalar< PScalar< PScalar < RScalar< Word< T > > > > >& a, DescribeLeaves& d)
    {
      d.add_literal< T >( a.shape() , a.get_word_value() );
      return 0;
    }
  };

  template<>
  struct LeafFunctor<GammaType, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const GammaType& g, DescribeLeaves& d)
    {
      d.add_gamma( g.elem() );
      return 0;
    }
  };

  template<>
  struct LeafFunctor<GammaTypeDP, DescribeLeaves>
  {
    typedef int Type_t;
    inline static Type_t apply(const GammaTypeDP& g, DescribeLeaves& d)
    {
      d.add_gamma( g.elem() );
      return 0;
    }
  };


#define QDP_PLAN_PEEK( FN , PARAMS )					\
  template<class A>							\
  struct ForEach<UnaryNode<FN, A>, DescribeLeaves, NullCombine>		\
  {									\
    typedef typename ForEach<A, DescribeLeaves, NullCombine>::Type_t Type_t; \
    inline static Type_t apply(const UnaryNode<FN, A>& expr, DescribeLeaves& d, const NullCombine& n) \
    {									\
      const FN& op = expr.operation();					\
      PARAMS;								\
      return ForEach<A, DescribeLeaves, NullCombine>::apply( expr.child() , d , n ); \
    }									\
  };

  QDP_PLAN_PEEK( FnPeekColorMatrix , d.add_op_param( op.getRow() ); d.add_op_param( op.getCol() ) )
  QDP_PLAN_PEEK( FnPeekSpinMatrix  , d.add_op_param( op.getRow() ); d.add_op_param( op.getCol() ) )
  QDP_PLAN_PEEK( FnPeekColorVector , d.add_op_param( op.getRow() ) )
  QDP_PLAN_PEEK( FnPeekSpinVector  , d.add_op_param( op.getRow() ) )
#undef QDP_PLAN_PEEK


  template<class A>
  struct ForEach<UnaryNode<FnMap, A>, DescribeLeaves, NullCombine>
  {
    typedef typename ForEach<A, DescribeLeaves, NullCombine>::Type_t Type_t;
    inline static Type_t apply(const UnaryNode<FnMap, A>& expr, DescribeLeaves& d, const NullCombine& n)
    {
      d.add_shift( expr.operation() , expr.operation().map.hasOffnode() );
      return ForEach<A, DescribeLeaves, NullCombine>::apply( expr.child() , d , n );
    }
  };


  template<class Op>
  struct AddOpDescribe
  {
    static void apply( const Op& , DescribeLeaves& ) {}
  };

  template<> struct AddOpDescribe<FnPokeColorMatrix> {
    static void apply( const FnPokeColorMatrix& op , DescribeLeaves& d ) { d.add_op_param( op.getRow() ); d.add_op_param( op.getCol() ); }
  };
  template<> struct AddOpDescribe<FnPokeSpinMatrix> {
    static void apply( const FnPokeSpinMatrix& op , DescribeLeaves& d ) { d.add_op_param( op.getRow() ); d.add_op_param( op.getCol() ); }
  };
  template<> struct AddOpDescribe<FnPokeColorVector> {
    static void apply( const FnPokeColorVector& op , DescribeLeaves& d ) { d.add_op_param( op.getRow() ); }
  };
  template<> struct AddOpDescribe<FnPokeSpinVector> {
    static void apply( const FnPokeSpinVector& op , DescribeLeaves& d ) { d.add_op_param( op.getRow() ); }
  };


  template<class T, class C1, class Op, class RHS>
  ExprPlan describe_expr( const OLattice<T>& dest , const Op& op , const QDPExpr<RHS,C1>& rhs )
  {
    ExprPlan plan;
    plan.entries.reserve( 32 );
    DescribeLeaves d( plan );
    forEach( dest , d , NullCombine() );
    AddOpDescribe<Op>::apply( op , d );
    forEach( rhs , d , NullCombine() );

    for ( size_t i = 1 ; i < plan.entries.size() ; ++i )
      {
	const PlanEntry& e = plan.entries[i];
	if ( e.kind == PlanEntry::Kind::Lattice && e.cache_id == plan.entries[0].cache_id )
	  plan.dest_overlap = true;
      }
    return plan;
  }


  template<class T, class C1, class Op, class RHS>
  ExprPlan describe_sca_sca( const OScalar<T>& dest , const Op& op , const QDPExpr<RHS,C1>& rhs )
  {
    ExprPlan plan;
    plan.entries.reserve( 16 );
    DescribeLeaves d( plan );
    d.add_scalar( dest.shape() , dest.get_ptr() , jit_sizeof( dest ) );
    AddOpDescribe<Op>::apply( op , d );
    forEach( rhs , d , NullCombine() );
    return plan;
  }


  template<class T>
  ExprPlan describe_lattice( const OLattice<T>& a )
  {
    ExprPlan plan;
    DescribeLeaves d( plan );
    forEach( a , d , NullCombine() );
    return plan;
  }


  template<class... Args>
  ExprPlan describe_leaves( const Args&... args )
  {
    ExprPlan plan;
    DescribeLeaves d( plan );
    ( forEach( args , d , NullCombine() ) , ... );
    return plan;
  }


  inline DynKey combined_key( const ExprPlan& a , const ExprPlan& b )
  {
    DynKey k = a.key();
    k.add( b.key() );
    return k;
  }

  inline std::vector<int> combined_structure( const ExprPlan& a , const ExprPlan& b )
  {
    std::vector<int> s = a.structure();
    std::vector<int> t = b.structure();
    s.insert( s.end() , t.begin() , t.end() );
    return s;
  }


  template<class C1, class RHS>
  ExprPlan describe_rhs( const QDPExpr<RHS,C1>& rhs , int layout_slot0 )
  {
    ExprPlan plan;
    plan.entries.reserve( 32 );
    plan.layout_of_slot.push_back( layout_slot0 );
    DescribeLeaves d( plan );
    forEach( rhs , d , NullCombine() );
    return plan;
  }



  //
  //
  inline void append_plan_args( std::vector<QDPCache::ArgKey>& ids , std::vector<int>& signoff ,
				const ExprPlan& plan , const Subset& subset ,
				std::vector<int>* written = nullptr )
  {
    QDPCache& cache = QDP_get_global_cache();
    const int layout_slot0 = plan.layout_of_slot.empty() ? -1 : plan.layout_of_slot[0];

    auto param = [&signoff]( int id ) { signoff.push_back( id ); return id; };

    std::vector<bool> slot_done( plan.layout_of_slot.size() , false );
    if ( !slot_done.empty() )
      slot_done[0] = true;

    for ( const PlanEntry& e : plan.entries )
      {
	switch (e.kind)
	  {
	  case PlanEntry::Kind::Lattice:
	    ids.push_back( e.cache_id );
	    if ( !slot_done[e.layout_slot] )
	      {
		ids.push_back( param( cache.addJitParamInt( Layout::from_index( e.layout_ref ).sitesOnNode() ) ) );
		ids.push_back( get_translation_sitetable_id( e.layout_ref , layout_slot0 ) );
		slot_done[e.layout_slot] = true;
	      }
	    break;
	  case PlanEntry::Kind::Scalar:
	    {
	      int id;
	      switch (e.mem)
		{
		case PlanEntry::ScalarMem::Ring:
		  id = QDP_get_global_ring_buffer().allocate( e.bytes , e.host_ptr );
		  break;
		case PlanEntry::ScalarMem::OwnHost:
		  id = param( cache.addOwnHostMem( e.bytes , e.host_ptr ) );
		  break;
		case PlanEntry::ScalarMem::OwnHostUndef:
		  id = param( cache.addOwnHostMemStatus( e.bytes , e.host_ptr , QDPCache::Status::undef ) );
		  break;
		}
	      if ( e.written )
		{
		  assert( written && "written scalar entry needs LaunchArgs" );
		  written->push_back( id );
		}
	      ids.push_back( id );
	    }
	    break;
	  case PlanEntry::Kind::Literal:
	    ids.push_back( param( std::visit( [&cache]( auto v ) {
		  typedef decltype(v) W;
		  if constexpr ( std::is_same_v<W,float> )        return cache.addJitParamFloat( v );
		  else if constexpr ( std::is_same_v<W,double> )  return cache.addJitParamDouble( v );
		  else if constexpr ( std::is_same_v<W,int> )     return cache.addJitParamInt( v );
		  else if constexpr ( std::is_same_v<W,int64_t> ) return cache.addJitParamInt64( v );
		  else                                            return cache.addJitParamBool( v );
		} , e.literal ) ) );
	    break;
	  case PlanEntry::Kind::OpParam:
	    ids.push_back( param( cache.addJitParamInt( e.op_param ) ) );
	    break;
	  case PlanEntry::Kind::Gamma:
	    break;
	  case PlanEntry::Kind::Shift:
	    ids.push_back( e.fnmap->map.getGoffsetsId( subset , layout_slot0 ) );
	    ids.push_back( e.offnode ? e.fnmap->getCached().getRecvBufId() : -1 );
	    break;
	  }
      }
  }


  //
  //
  class LaunchArgs
  {
    std::vector<QDPCache::ArgKey> ids_;
    std::vector<int> signoff;
    std::vector<int> written;
    int pos_th_count  = -1;
    int pos_sitetable = -1;

    int param( int id )
    {
      signoff.push_back( id );
      return id;
    }

  public:
    LaunchArgs() = default;

    explicit LaunchArgs( const ExprPlan& plan )
    {
      ids_.reserve( 3 + plan.entries.size() + 2 * plan.layout_of_slot.size() );
    }

    void add_iteration( int th_count , int latsize )
    {
      QDPCache& cache = QDP_get_global_cache();
      pos_th_count = ids_.size();
      ids_.push_back( param( cache.addJitParamInt( th_count ) ) );
      ids_.push_back( param( cache.addJitParamInt( latsize ) ) );
    }

    void add_iteration( int th_count , int latsize , int sitetable_id )
    {
      add_iteration( th_count , latsize );
      pos_sitetable = ids_.size();
      ids_.push_back( sitetable_id );
    }

    void add_int( int value )   { ids_.push_back( param( QDP_get_global_cache().addJitParamInt( value ) ) ); }
    void add_bool( bool value ) { ids_.push_back( param( QDP_get_global_cache().addJitParamBool( value ) ) ); }

    void add_id( int id ) { ids_.push_back( id ); }

    void add_plan( const ExprPlan& plan , const Subset& subset )
    {
      append_plan_args( ids_ , signoff , plan , subset , &written );
    }

    //! iteration space (e.g. the face sites after the inner sites)
    void reset_iteration( int th_count , int sitetable_id )
    {
      assert( pos_sitetable >= 0 );
      ids_[ pos_th_count ]  = param( QDP_get_global_cache().addJitParamInt( th_count ) );
      ids_[ pos_sitetable ] = sitetable_id;
    }

    void launch( JitFunction& function , int th_count , bool set_l0_event = false )
    {
      jit_launch( function , th_count , ids_ , set_l0_event );
    }

    bool launch( JitFunction& function , const kernel_geom_t& geom , size_t shared )
    {
      return jit_launch_explicit_geom( function , ids_ , geom , shared );
    }

    //! After the launch: copy the scalars the kernel wrote to the host
    void copy_written_to_host()
    {
      for ( int id : written )
	QDP_get_global_cache().assureOnHost( id );
      written.clear();
    }

    std::vector<QDPCache::ArgKey>& ids() { return ids_; }

    ~LaunchArgs()
    {
      assert( written.empty() && "copy_written_to_host() not called" );
      for ( int id : signoff )
	QDP_get_global_cache().signoff( id );
    }

    LaunchArgs( const LaunchArgs& ) = delete;
    LaunchArgs& operator=( const LaunchArgs& ) = delete;
  };


  //
  //
  class KernelBuilder
  {
    struct Start
    {
      Start( const char* name , const char* pretty ) { llvm_start_new_function( name , pretty ); }
    };

    Start           start;
    const ExprPlan& plan;
    IterationSpace  iter;
    MGParamLeaf     leaves;
    int             num_params;

  public:
    KernelBuilder( const char* name , const char* pretty , const ExprPlan& plan_ , bool with_sitetable = true ):
      start( name , pretty ), plan( plan_ ), iter( with_sitetable ), leaves( plan_ , iter.latsize_dest() ),
      num_params( with_sitetable ? 3 : 2 )
    {}

    template<class T>
    auto view( const T& x )
    {
      return forEach( x , leaves , TreeCombine() );
    }

    template<class T>
    ParamRef add_param()
    {
      ++num_params;
      return llvm_add_param<T>();
    }

    llvm::Value* site() { return iter.get_idx_dest(); }

    llvm::Value* thread() { return iter.get_idx_thread(); }

    ParamRef latsize() const { return iter.latsize_dest(); }

    MGParamLeaf& param_leaf() { return leaves; }

    void finish( JitFunction& function )
    {
      assert( leaves.all_entries_used() );
      jit_get_function( function );
      assert( function.get_num_params() < 0 ||
	      function.get_num_params() == num_params + plan.num_leaf_params() );
    }
  };


}

#endif
