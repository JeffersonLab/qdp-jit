#ifndef QDP_MAP_H
#define QDP_MAP_H

/*! @file
 * @brief Map classes
 *
 * Support classes for maps/shifts
 */

//#include <signal.h>

namespace QDP {

// Helpful for communications
#define FORWARD 1
#define BACKWARD -1


/*! @defgroup map Maps and shifts
 *
 * Maps are the mechanism for communications. Under a map,
 * a data-parallel object is mapped uniquely from sites to
 * sites. Nearest neighbor shifts are an example of the more
 * generic map.
 *
 * @{
 */

//! MapFunc 
/*! Abstract base class used as a function object for constructing maps */
class MapFunc
{
public:
  //! Virtual destructor - no cleanup needed
  virtual ~MapFunc() {}
  //! Maps a lattice coordinate under a map to a new lattice coordinate
  /*! sign > 0 for map, sign < 0 for the inverse map */
  virtual multi1d<int> operator() (const multi1d<int>& coordinate, int sign) const = 0;
};
    

//! ArrayMapFunc 
/*! Abstract base class used as a function object for constructing maps */
class ArrayMapFunc
{
public:
  //! Virtual destructor - no cleanup needed
  virtual ~ArrayMapFunc() {}

  //! Maps a lattice coordinate under a map to a new lattice coordinate
  /*! sign > 0 for map, sign < 0 for the inverse map */
  virtual multi1d<int> operator() (const multi1d<int>& coordinate, int sign, int dir) const = 0;

  //! Returns the array size - the number of directions which are to be used
  virtual int numArray() const = 0;
};


    
/** @} */ // end of group map


struct FnMap
{
  //PETE_EMPTY_CONSTRUCTORS(FnMap)
private:
  FnMap& operator=(const FnMap& f);

public:
  const Map& map;
  shared_ptr<RsrcWrapper> pRsrc;

  FnMap(const Map& m);
  FnMap(const FnMap& f);
  void init_wrapper(int layout_ref);
  
  const FnMapRsrc& getResource(int srcnum_, int dstnum_) {
    //assert(pRsrc);
    return pRsrc->getResource( srcnum_ , dstnum_ );
  }

  const FnMapRsrc& getCached() const {
    //assert(pRsrc);
    return pRsrc->get();
  }
  
  template<class T>
  inline typename UnaryReturn<T, FnMap>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }

};



template<class T>
struct HasShift
{
  constexpr static bool value = false;
};

template<class Op, class A, class B>
struct HasShift< BinaryNode<Op,A,B> >
{
  constexpr static bool value = HasShift<A>::value || HasShift<B>::value;
};

template<class Op, class A, class B, class C>
struct HasShift< TrinaryNode<Op,A,B,C> >
{
  constexpr static bool value = HasShift<A>::value || HasShift<B>::value || HasShift<C>::value;
};

template<class Op, class A>
struct HasShift< UnaryNode<Op,A> >
{
  constexpr static bool value = HasShift<A>::value;
};

template<class A>
struct HasShift< UnaryNode<FnMap,A> >
{
  constexpr static bool value = true;
};

template<typename T> concept ConceptHasShift   =  HasShift<T>::value;
template<typename T> concept ConceptHasNoShift = !HasShift<T>::value;



//! General permutation map class for communications
class Map
{
public:
  //! Constructor - does nothing really
  Map() {}

  ~Map();

  //! Constructor from a function object
  Map(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign) */
  void make(const MapFunc& func);


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(*this),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(*this),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


  void make_lazy(const Subset& s, int ref) const;

  const multi1d<int>& goffset(const Subset& s, int ref) const { return done( s , ref , "goffset" ).goffsets; }
  const multi1d<int>& soffset(const Subset& s, int ref) const { return done( s , ref , "soffset" ).soffsets; }
  const multi1d<int>& roffset(const Subset& s, int ref) const { return done( s , ref , "roffset" ).roffsets; }

  int getRoffsetsId   (const Subset& s, int ref) const { return lazy(s,ref).roffsetsId; }
  int getSoffsetsId   (const Subset& s, int ref) const { return lazy(s,ref).soffsetsId; }
  int getGoffsetsId   (const Subset& s, int ref) const { return lazy(s,ref).goffsetsId; }
  int get_destnode_num(const Subset& s, int ref) const { return lazy(s,ref).destnode_num; }
  int get_srcenode_num(const Subset& s, int ref) const { return lazy(s,ref).srcenode_num; }

  int getId() const {
    if (!offnodeP)
      return 0;
    if (myId < 0)
      {
	QDPIO::cout << "internal error. Map::getId called before lazy evaluation for any subset\n";
	raise(SIGSEGV);
	QDP_error_exit("giving up");
      }
    return myId;
  }
  bool get_offnodeP() const { return offnodeP; }
  bool hasOffnode() const   { return offnodeP; }

  int get_destnode(int ref) const { return layout( ref , "get_destnode" ).destnode; }
  int get_srcenode(int ref) const { return layout( ref , "get_srcenode" ).srcenode; }

  bool lazy_done_s(const Subset& s, int ref) const {
    const int id = s.getId();
    if (id < 0)
      {
	QDPIO::cerr << "Map: subset without id (not part of a set)" << std::endl;
	QDP_abort(1);
      }
    auto it = layouts.find( ref );
    if ( it == layouts.end() )
      return false;
    const std::deque<subset_t>& subsets = it->second.subsets;
    return id < (int)subsets.size() && subsets[id].done &&
      subsets[id].generation == MasterSet::Instance().generation( id );
  }


  //! Hide copy constructor
  Map(const Map&) = delete;

  //! Hide assignment operator
  Map& operator=(const Map&) = delete;

private:
  struct subset_t
  {
    multi1d<int> goffsets;
    multi1d<int> soffsets;
    multi1d<int> roffsets;

    int goffsetsId = -1;
    int soffsetsId = -1;
    int roffsetsId = -1;

    int srcenode_num = 0;
    int destnode_num = 0;

    bool done = false;
    int  generation = 0;
  };

  struct map_t
  {
    std::deque<subset_t> subsets;

    // LAZY
    multi1d< multi1d<int> > lazy_fcoord;
    multi1d< multi1d<int> > lazy_bcoord;
    multi1d<int>    srcnode;
    multi1d< multi1d<int> > lazy_destnode_fcoord;

    int srcenode;
    int destnode;
  };


  const subset_t& lazy(const Subset& s, int ref) const
  {
    if (!lazy_done_s(s,ref))
      make_lazy(s,ref);
    return layouts[ ref ].subsets[ s.getId() ];
  }

  const subset_t& done(const Subset& s, int ref, const char* what) const
  {
    if (!lazy_done_s(s,ref))
      {
	QDPIO::cerr << "Map: " << what << " used before the lazy component was built" << std::endl;
	QDP_abort(1);
      }
    return layouts[ ref ].subsets[ s.getId() ];
  }

  const map_t& layout( int ref , const char* what ) const
  {
    auto it = layouts.find( ref );
    if ( it == layouts.end() )
      {
	QDPIO::cerr << "Map " << what << ": map not made for layout " << ref << std::endl;
	QDP_abort(1);
      }
    return it->second;
  }

  void signoff_lazy( subset_t& l ) const;


  mutable std::map< int , map_t > layouts;
  
  mutable int myId = -1; // master map id

  // Indicate off-node communications is needed;
  bool offnodeP = false;
};






namespace Gather {
  template<class T1, class RHS>
  void run( int send_buf_id , const Map& map , const QDPExpr<RHS,OLattice<T1> >& rhs , const Subset& subset , int layout_ref , bool set_l0_event );
}


struct FnMapJIT
{
  IndexRet index;
  const Map& map;
  shared_ptr<RsrcWrapper> pRsrc;

  FnMapJIT(const FnMap& fnmap,const IndexRet& i): 
    index(i), map(fnmap.map), pRsrc(fnmap.pRsrc)  {}
  FnMapJIT(const FnMapJIT& f) : index(f.index), map(f.map) , pRsrc(f.pRsrc)  {}

  template<class T>
  inline typename UnaryReturn<T, FnMapJIT>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }
};


















template<class A>
struct ForEach<UnaryNode<FnMap, A>, MGParamLeaf, TreeCombine>
{
  typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t InnerType_t;
  typedef typename ForEach< UnaryNode<FnMapJIT, A> , MGParamLeaf, TreeCombine>::Type_t Type_t;
  inline
  static Type_t apply(const UnaryNode<FnMap, A>& expr, MGParamLeaf &p, const TreeCombine &c)
  {
    typedef typename WordType<InnerType_t>::Type_t AWordType_t;

    const PlanEntry& e = p.next( PlanEntry::Kind::Shift );

    IndexRet index_pack;
    index_pack.p_multi_index = llvm_add_param<int*>();
    index_pack.p_recv_buf    = llvm_add_param<AWordType_t*>();
    index_pack.offnode       = e.offnode != 0;

    return Type_t( FnMapJIT( expr.operation() , index_pack ) , 
		   ForEach< A , MGParamLeaf , TreeCombine >::apply( expr.child() , p , c ) );
  }
};


template<class A> struct IsLatticeLeafJIT : std::false_type {};
template<class T> struct IsLatticeLeafJIT< OLatticeJIT<T> > : std::true_type {};
template<class T> struct IsLatticeLeafJIT< Reference< OLatticeJIT<T> > > : std::true_type {};


template<class A>
struct ForEach<UnaryNode<FnMapJIT, A>, MGViewLeaf, OpCombine>
{
  typedef typename ForEach<A, MGViewLeaf, OpCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMapJIT , OpCombine>::Type_t Type_t; // This is a JIT container (now)
  inline
  static Type_t apply(const UnaryNode<FnMapJIT, A>& expr, MGViewLeaf &v, const OpCombine &o)
  {
    IndexRet index = expr.operation().index;

    llvm::Value * r_multi_index = llvm_array_type_indirection<int>( index.p_multi_index , v.getIndex( 0 ) );

    if ( IsLatticeLeafJIT<A>::value && !index.offnode )
      {
	MGViewLeaf vv( r_multi_index , v.param_leaf() );
	return Combine1<TypeA_t, 
			FnMapJIT , 
			OpCombine>::combine(ForEach<A, MGViewLeaf, OpCombine>::apply(expr.child(), vv, o) , 
					    expr.operation(), o);
      }

    DummyJitLeaf jitleaf;
    Shape shape = forEach( expr.child() , jitleaf , OpEvalShape() ).shape();
	  
    Type_t ret = stack_alloc_jit< Type_t >(shape);
	  
    JitIf inRecvBuffer( llvm_lt( r_multi_index , llvm_create_value(0) ) );
    {
      llvm::Value *idx_buf = llvm_sub ( llvm_neg ( r_multi_index ) , llvm_create_value(1) );

      NestedIndex args;
      args.push_back( llvm_create_value( Layout::sitesOnNode() ) , idx_buf );
      args.push_back( llvm_create_value(1) , llvm_create_value(0) );

      Type_t t_jit_recv( llvm_derefParam(index.p_recv_buf) , JitDataOrder::SiteSlow ,args , shape );

      ret = t_jit_recv;
    }
    inRecvBuffer.els();
    {
      MGViewLeaf vv( r_multi_index , v.param_leaf() );
      ret = Combine1<TypeA_t, 
		     FnMapJIT , 
		     OpCombine>::combine(ForEach<A, MGViewLeaf, OpCombine>::apply(expr.child(), vv, o) , 
					 expr.operation(), o);
    }
    inRecvBuffer.end();

    return ret;
  }
};













template<class A>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase1 , BitOrCombine>
{
  typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t InnerTypeA_t;
  typedef typename Combine1<InnerTypeA_t, FnMap, OpCombine>::Type_t InnerType_t;
  typedef int Type_t;
  typedef QDPExpr<A,OLattice<InnerType_t> > Expr;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, ShiftPhase1 &f, const BitOrCombine &c)
  {
    const Map& map = expr.operation().map;

    FnMap& fnmap = const_cast<FnMap&>(expr.operation());
    
    Expr subexpr(expr.child());

    if (f.in_shift)
      {
	QDPIO::cerr << "shift of shift is not supported" << std::endl;
	QDP_abort(1);
      }
    f.in_shift = true;
    forEach(subexpr, f , BitOrCombine());
    f.in_shift = false;

    if (map.get_offnodeP())
      {
#if QDP_DEBUG >= 3
	QDP_info("Map: off-node communications required");
#endif
	fnmap.init_wrapper( f.layout_ref );

	Shape shape = shape_of( subexpr );
	//std::cout << "shift phase 1: shape = " << shape;
	OScalar<InnerType_t> dummy(shape);

	int size_inner = jit_sizeof(dummy);
	
	int dstnum = map.get_destnode_num(f.subset,f.layout_ref) * size_inner;
	int srcnum = map.get_srcenode_num(f.subset,f.layout_ref) * size_inner;

	const FnMapRsrc& rRSrc = fnmap.getResource(srcnum,dstnum);

#ifdef QDP_BACKEND_L0
	Gather::run( rRSrc.getSendBufId() , map , subexpr , f.subset , f.layout_ref , jit_config_get_gpu_direct() );
	if ( jit_config_get_gpu_direct() )
	  {
	    gpu_wait_l0_event();
	  }
#else
	Gather::run( rRSrc.getSendBufId() , map , subexpr , f.subset , f.layout_ref , false );

	if ( jit_config_get_gpu_direct() )
	  {
	    gpu_sync();
	  }
#endif
	
	rRSrc.send_receive();

      }
    map.make_lazy(f.subset,f.layout_ref);
    

    return map.getId();
  }
};




template<class A>
struct ForEach<UnaryNode<FnMap, A>, SelfAssignTag , BitOrCombine>
{
  typedef typename ForEach<A, SelfAssignTag, BitOrCombine>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, SelfAssignTag &f, const BitOrCombine &c)
  {
    ++f.shift_depth;
    Type_t ret = ForEach<A, SelfAssignTag, BitOrCombine>::apply(expr.child(), f, c);
    --f.shift_depth;
    return ret;
  }
};



template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase2 , CTag>
{
  typedef int Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, ShiftPhase2 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());
    if (map.get_offnodeP()) {
      const FnMapRsrc& rRSrc = fnmap.getCached();
      rRSrc.qmp_wait();
    }
    return ForEach<A, ShiftPhase2, CTag>::apply(expr.child(), f, c);
  }
};





//-----------------------------------------------------------------------------
//! Array of general permutation map class for communications
class ArrayMap
{
public:
  //! Constructor - does nothing really
  ArrayMap() {}

  //! Destructor
  ~ArrayMap() {}

  //! Constructor from a function object
  ArrayMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign,dir) */
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,dir)
   *
   * Implements:  dest(x) = source(map(x,dir))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir]),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir]),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
  ArrayMap(const ArrayMap&) {}

  //! Hide operator=
  void operator=(const ArrayMap&) {}

private:
  multi1d<Map> mapsa;
  
};

//-----------------------------------------------------------------------------
//! BiDirectional of general permutation map class for communications
class BiDirectionalMap
{
public:
  //! Constructor - does nothing really
  BiDirectionalMap() {}

  //! Destructor
  ~BiDirectionalMap() {}

  //! Constructor from a function object
  BiDirectionalMap(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign) */
  void make(const MapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,isign)
   *
   * Implements:  dest(x) = source(map(x,isign))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */

  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1]),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1]),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
  BiDirectionalMap(const BiDirectionalMap&) {}

  //! Hide operator=
  void operator=(const BiDirectionalMap&) {}

private:
  multi1d<Map> bimaps;
  
};


//-----------------------------------------------------------------------------
//! ArrayBiDirectional of general permutation map class for communications
class ArrayBiDirectionalMap
{
public:
  //! Constructor - does nothing really
  ArrayBiDirectionalMap() {}

  //! Destructor
  ~ArrayBiDirectionalMap() {}

  //! Constructor from a function object
  ArrayBiDirectionalMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign,dir) */
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * Implements:  dest(x) = source(map(x,isign,dir))
   *
   * Syntax:
   * map(source,isign,dir)
   *
   * isign = parity of direction (+1 or -1)
   * dir   = array index (could be direction in range [0,...,Nd-1])
   *
   * Implements:  dest(x) = s1(x+isign*dir)
   * There are cpp macros called  FORWARD and BACKWARD that are +1,-1 resp.
   * that are often used as arguments
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir)),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir)),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
  ArrayBiDirectionalMap(const ArrayBiDirectionalMap&) {}

  //! Hide operator=
  void operator=(const ArrayBiDirectionalMap&) {}

private:
  multi2d<Map> bimapsa;
  
};




} // namespace QDP

#endif
