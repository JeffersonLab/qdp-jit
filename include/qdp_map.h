// -*- C++ -*-

/*! @file
 * @brief Map classes
 *
 * Support classes for maps/shifts
 */

#ifndef QDP_MAP_H
#define QDP_MAP_H

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
  //std::shared_ptr<RsrcWrapper> pRsrc;
  QDPHandle::Handle<RsrcWrapper> pRsrc;

  FnMap(const Map& m);
  FnMap(const FnMap& f);

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



//! General permutation map class for communications
class Map
{
public:
  //! Constructor - does nothing really
  Map() {}

  //! Destructor
  ~Map() {}

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


public:
  void make_lazy(const Subset& s) const;
  //! Accessor to offsets
  const multi1d<int>& goffset(const Subset& s) const {
    if (!lazy_done_s(s))
      QDP_error_exit("goffest used before lazy component was called");
    assert( s.getId() >= 0 && s.getId() < goffsets.size() && "goffset: subset Id out of range");
    return goffsets[s.getId()];
  }
  const multi1d<int>& soffset(const Subset& s) const {
    if (!lazy_done_s(s))
      QDP_error_exit("soffest used before lazy component was called");
    assert( s.getId() >= 0 && s.getId() < soffsets.size() && "soffset: subset Id out of range");
    return soffsets[s.getId()];
  }
  const multi1d<int>& roffset(const Subset& s) const {
    if (!lazy_done_s(s))
      QDP_error_exit("roffest used before lazy component was called");
    assert( s.getId() >= 0 && s.getId() < roffsets.size() && "roffset: subset Id out of range");
    return roffsets[s.getId()];
  }
  int getRoffsetsId(const Subset& s) const { 
    if (!lazy_done_s(s))
      make_lazy(s);
    assert( s.getId() >= 0 && s.getId() < roffsets.size() && "roffset: subset Id out of range");
    return roffsetsId[s.getId()];
  }
  int getSoffsetsId(const Subset& s) const { 
    if (!lazy_done_s(s))
      make_lazy(s);
    assert( s.getId() >= 0 && s.getId() < soffsets.size() && "soffset: subset Id out of range");
    return soffsetsId[s.getId()];
  }
  int getGoffsetsId(const Subset& s) const { 
    if (!lazy_done_s(s))
      make_lazy(s);
    assert( s.getId() >= 0 && s.getId() < goffsets.size() && "goffset: subset Id out of range");
    return goffsetsId[s.getId()];
  }
  const multi1d<int>& get_destnodes_num(const Subset& s) const {
    if (!lazy_done_s(s))
      make_lazy(s);
    return destnodes_num[s.getId()];
  }
  const multi1d<int>& get_srcenodes_num(const Subset& s) const {
    if (!lazy_done_s(s))
      make_lazy(s);
    return srcenodes_num[s.getId()];
  }
  int getId() const {
    if (myId < 0)
      {
	QDPIO::cout << "internal error. Map::getId called before lazy evaluation for any subset\n";
	QDP_error_exit("giving up");
      }
    return myId;
  }
  bool get_offnodeP() const { return offnodeP; }
  bool hasOffnode() const   { return offnodeP; }
  const multi1d<int>& get_destnodes() const {
    return destnodes;
  }
  const multi1d<int>& get_srcenodes() const {
    return srcenodes;
  }
  bool lazy_done_s(const Subset& s) const {
    if (lazy_done.size() > 0  &&  lazy_done.size() < s.getId() ) {
      QDPIO::cout << "subset Id out of range. Did you use shift on a user-defined subset?\n";
      QDP_error_exit("giving up");
    }
    if (lazy_done.size() < 1)
      return false;
    return lazy_done[ s.getId() ];
  }


  const multi1d< multi1d<int> >& get_goffsets() const { return goffsets; }
  multi1d< multi1d<int> >& get_goffsets() { return goffsets; }


private:
  //! Hide copy constructor
  Map(const Map&) {}

  //! Hide operator=
  void operator=(const Map&) {}

private:
  mutable multi1d< multi1d<int> > goffsets;    // [subset no.][linear index] > 0 local, < 0 receive buffer index
  mutable multi1d< multi1d<int> > soffsets;    // [subset no.][0..N] = linear index   N = destnodes_num
  mutable multi1d< multi1d<int> > roffsets;    // [subset no.][0..N] = linear index   N = srcenodes_num

  mutable multi1d<int> roffsetsId; // [subset no.]
  mutable multi1d<int> soffsetsId; // [subset no.]
  mutable multi1d<int> goffsetsId; // [subset no.]

  mutable int myId = -1; // master map id

  multi1d<int> srcenodes;                   // node number index = node number
  multi1d<int> destnodes;                   // node number index = node number

  mutable multi1d< multi1d<int> > srcenodes_num;    // [subset no.][node number index] = number of sites
  mutable multi1d< multi1d<int> > destnodes_num;    // [subset no.][node number index] = number of sites

  // Indicate off-node communications is needed;
  bool offnodeP;


  // LAZY
  multi1d< multi1d<int> > lazy_fcoord;
  multi1d< multi1d<int> > lazy_bcoord;
  mutable multi1d<int>    srcnode;
  multi1d< multi1d<int> > lazy_destnodes0_fcoord;
  mutable multi1d<bool>   lazy_done;                  // [subset no.]
};






struct FnMapJIT
{
public:
  IndexRet index;
  const Map& map;
  //std::shared_ptr<RsrcWrapper> pRsrc;
  QDPHandle::Handle<RsrcWrapper> pRsrc;

  FnMapJIT(const FnMap& fnmap,const IndexRet& i): 
    index(i), map(fnmap.map), pRsrc(fnmap.pRsrc)  {}
  FnMapJIT(const FnMapJIT& f) : index(f.index), map(f.map) , pRsrc(f.pRsrc)  {}

public:
  template<class T>
  inline typename UnaryReturn<T, FnMapJIT>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }
};



#if defined(QDP_USE_PROFILING)   
template <>
struct TagVisitor<FnMap, PrintTag> : public ParenPrinter<FnMap>
{ 
  static void visit(FnMap op, PrintTag t) 
    { t.os_m << "shift"; }
};
#endif





template<class A>
struct ForEach<UnaryNode<FnMap, A>, ParamLeaf, TreeCombine>
  {
    typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t InnerTypeA_t;
    typedef typename Combine1<InnerTypeA_t, FnMap, OpCombine>::Type_t InnerType_t;
    //typedef typename ForEach<A, ViewLeaf, OpCombine>::Type_t AInnerTypeA_t;
    typedef typename ForEach< UnaryNode<FnMapJIT, A> , ParamLeaf, TreeCombine>::Type_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnMap, A>& expr, const ParamLeaf &p, const TreeCombine &c)
    {
      //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

      //expr.operation().map;
      //FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      typedef typename WordType<InnerType_t>::Type_t AWordType_t;

      IndexRet index_pack;
      index_pack.p_multi_index = llvm_add_param<int*>();
      index_pack.p_recv_buf    = llvm_add_param<AWordType_t*>(); // This deduces it's type from A

      return Type_t( FnMapJIT( expr.operation() , index_pack ) , 
		     ForEach< A, ParamLeaf, TreeCombine >::apply( expr.child() , p , c ) );
    }
  };





template<class A>
struct ForEach<UnaryNode<FnMapJIT, A>, ViewLeaf, OpCombine>
  {
    typedef typename ForEach<A, ViewLeaf, OpCombine>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnMapJIT , OpCombine>::Type_t Type_t; // This is a REG container
    inline
    static Type_t apply(const UnaryNode<FnMapJIT, A>& expr, const ViewLeaf &v, const OpCombine &o)
    {
      Type_t ret;
      Type_t ret_phi0;
      Type_t ret_phi1;

      IndexRet index = expr.operation().index;

      llvm::Value * r_multi_index = llvm_array_type_indirection( index.p_multi_index , v.getIndex() );
      
      llvm::BasicBlock * block_in_buffer = llvm_new_basic_block();
      llvm::BasicBlock * block_not_in_buffer = llvm_new_basic_block();
      llvm::BasicBlock * block_in_buffer_exit = llvm_new_basic_block();
      llvm_cond_branch( llvm_lt( r_multi_index , 
				 llvm_create_value(0) ) , 
			block_in_buffer , 
			block_not_in_buffer );
      {
	llvm_set_insert_point(block_in_buffer);
	llvm::Value *idx_buf = llvm_sub ( llvm_neg ( r_multi_index ) , llvm_create_value(1) );

	IndexDomainVector args;
	args.push_back( make_pair( Layout::sitesOnNode() , idx_buf ) );
	args.push_back( make_pair( 1 , llvm_create_value(0) ) );

	typename JITType<Type_t>::Type_t t_jit_recv;
	t_jit_recv.setup( llvm_derefParam(index.p_recv_buf) ,
			  JitDeviceLayout::Scalar ,
			  args );

	ret_phi0.setup( t_jit_recv );
	
	llvm_branch( block_in_buffer_exit );
      }
      {
	llvm_set_insert_point(block_not_in_buffer);

	ViewLeaf vv( JitDeviceLayout::Coalesced , r_multi_index );
	ret_phi1 = Combine1<TypeA_t, 
			    FnMapJIT , 
			    OpCombine>::combine(ForEach<A, ViewLeaf, OpCombine>::apply(expr.child(), vv, o) , 
						expr.operation(), o);

	llvm_branch( block_in_buffer_exit );
      }
      llvm_set_insert_point(block_in_buffer_exit);

      qdpPHI( ret , 
	      ret_phi0 , block_in_buffer ,
	      ret_phi1 , block_not_in_buffer );

      return ret;
    }
  };






template<class A>
struct ForEach<UnaryNode<FnMap, A>, AddressLeaf, NullCombine>
  {
    typedef typename ForEach< A , AddressLeaf, NullCombine>::Type_t TypeA_t;
    typedef TypeA_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnMap, A>& expr, const AddressLeaf &a, const NullCombine &n)
    {
      const Map& map = expr.operation().map;
      FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      // int goffsetsId = expr.operation().map.getGoffsetsId(a.subset);
      // void * goffsetsDev = QDP_get_global_cache().getDevicePtr( goffsetsId );
      // a.setAddr( goffsetsDev );
      
      a.setId( expr.operation().map.getGoffsetsId(a.subset) );

      a.setId( map.hasOffnode() ? fnmap.getCached().getRecvBufId() : -1 );

      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
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
  Type_t apply(const UnaryNode<FnMap, A> &expr, const ShiftPhase1 &f, const BitOrCombine &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());

    //const int nodeSites = Layout::sitesOnNode();
    int returnVal=0;

    Expr subexpr(expr.child());

    if (map.get_offnodeP())
      {
#if QDP_DEBUG >= 3
	QDP_info("Map: off-node communications required");
#endif

	int dstnum = map.get_destnodes_num(f.subset)[0]*sizeof(InnerType_t);
	int srcnum = map.get_srcenodes_num(f.subset)[0]*sizeof(InnerType_t);

	const FnMapRsrc& rRSrc = fnmap.getResource(srcnum,dstnum);

	// Make sure the inner expression's map function
	// send and receive before recursing down
	int maps_involved = forEach(subexpr, f , BitOrCombine());
	if (maps_involved > 0) {
	  QDP_error_exit("shift of shift is not supported");
	  // ShiftPhase2 phase2;
	  // forEach(subexpr, phase2 , NullCombine());
	}

	static CUfunction function;

	if (function == NULL)
	  {
	    function = function_gather_build<InnerType_t>( subexpr );
	  }

	function_gather_exec(function, rRSrc.getSendBufId() , map , subexpr , f.subset );

	rRSrc.send_receive();
	
	returnVal = maps_involved | map.getId();
      }
    else 
      {
	returnVal = ForEach<A, ShiftPhase1, BitOrCombine>::apply(expr.child(), f, c);
      }
    return returnVal;
  }
};




template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase2 , CTag>
{
  typedef int Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const ShiftPhase2 &f, const CTag &c)
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


#if 0
// Add this code if you need CPU shifts
template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalLeaf1, CTag>
{
  typedef typename ForEach<A, EvalLeaf1, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalLeaf1 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());
    
    //EvalLeaf1 ff( map.goffsets[f.val1()] );
    EvalLeaf1 ff( map.get_goffsets()[f.val1()] );
    return Combine1<TypeA_t, FnMap, CTag>::combine(ForEach<A, EvalLeaf1, CTag>::apply(expr.child(), ff, c),expr.operation(), c);
  }
};
#endif



} // namespace QDP

#endif
