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

  //#define JIT_TIMING

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
  //! Accessor to offsets
  const multi1d<int>& goffset(const Subset& s) const {
    assert( s.getId() >= 0 && s.getId() < goffsets.size() && "goffset: subset Id out of range");
    return goffsets[s.getId()];
  }
  const multi1d<int>& soffset(const Subset& s) const {
    assert( s.getId() >= 0 && s.getId() < soffsets.size() && "soffset: subset Id out of range");
    return soffsets[s.getId()];
  }
  multi1d<int>& soffset(const Subset& s) {
    assert( s.getId() >= 0 && s.getId() < soffsets.size() && "soffset: subset Id out of range");
    return soffsets[s.getId()];
  }
  const multi1d<int>& roffset(const Subset& s) const {
    assert( s.getId() >= 0 && s.getId() < roffsets.size() && "roffset: subset Id out of range");
    return roffsets[s.getId()];
  }

  const multi1d< multi1d<int> >& get_srcenodes_num() const { return srcenodes_num; }
  const multi1d< multi1d<int> >& get_destnodes_num() const { return destnodes_num; }

  const multi1d<int>& get_srcenodes() const { return srcenodes; }
  const multi1d<int>& get_destnodes() const { return destnodes; }

  //multi1d<int>& soffset() {return soffsets;}

  // int getRoffsetsId() const { return roffsetsId;}
  // int getSoffsetsId() const { return soffsetsId;}
  // int getGoffsetsId() const { return goffsetsId;}

  int getId() const {return myId;}
  bool hasOffnode() const { return offnodeP; }

private:
  //! Hide copy constructor
  Map(const Map&) {}

  //! Hide operator=
  void operator=(const Map&) {}

private:
  friend class FnMap;
  friend class FnMapRsrc;
  template<class E,class F,class C> friend class ForEach;

  //! Offset table used for communications. 
  /*! 
   * The direction is in the sense of the Map or Shift functions from QDP.
   * goffsets(position) 
   */ 
  multi1d< multi1d<int> > goffsets;    // [subset no.][linear index] > 0 local, < 0 receive buffer index
  multi1d< multi1d<int> > soffsets;    // [subset no.][0..N] = linear index   N = destnodes_num
  multi1d< multi1d<int> > roffsets;    // [subset no.][0..N] = linear index   N = srcenodes_num

  // int roffsetsId;
  // int soffsetsId;
  // int goffsetsId;
  int myId; // master map id

  multi1d<int> srcenodes;                   // node number index = node number
  multi1d<int> destnodes;                   // node number index = node number

  multi1d< multi1d<int> > srcenodes_num;    // [subset no.][node number index] = number of sites
  multi1d< multi1d<int> > destnodes_num;    // [subset no.][node number index] = number of sites

  // Indicate off-node communications is needed;
  bool offnodeP;
};






struct FnMapJIT
{
public:
  IndexRet index;
  const Map& map;
  //std::shared_ptr<RsrcWrapper> pRsrc;
  QDPHandle::Handle<RsrcWrapper> pRsrc;

  FnMapJIT(const FnMap& fnmap,const IndexRet& i): 
    map(fnmap.map), pRsrc(fnmap.pRsrc), index(i) {}
  FnMapJIT(const FnMapJIT& f) : map(f.map) , pRsrc(f.pRsrc), index(f.index) {}

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

      const Map& map = expr.operation().map;
      FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      typedef typename WordType<InnerType_t>::Type_t AWordType_t;

      // if (llvm_debug::debug_func_write && Layout::primaryNode()) {
      // 	std::cout << "site permutation buffer\n";
      // 	std::cout << "receive buffer\n";
      // }

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
    //typedef typename ForEach< UnaryNode<FnMapJIT, A> , ParamLeaf, TreeCombine>::Type_t Type_t;
    typedef typename ForEach<A, ViewLeaf, OpCombine>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnMapJIT , OpCombine>::Type_t Type_t; // This is a REG container
    //typedef typename REGType< Type_t >::Type_t REGType_t;
    inline
    static Type_t apply(const UnaryNode<FnMapJIT, A>& expr, const ViewLeaf &v, const OpCombine &o)
    {
      //assert(!"ni");
#if 1
      Type_t ret;
      Type_t ret_phi0;
      Type_t ret_phi1;

      IndexRet index = expr.operation().index;

      llvm::Value * r_multi_index = llvm_array_type_indirection( index.p_multi_index , v.getIndex() );
      
      llvm::BasicBlock * block_in_buffer = llvm_new_basic_block();
      llvm::BasicBlock * block_not_in_buffer = llvm_new_basic_block();
      llvm::BasicBlock * block_in_buffer_exit = llvm_new_basic_block();
      llvm::BasicBlock * cond_exit;
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
			  JitDeviceLayout::LayoutScalar ,
			  args );

	ret_phi0.setup( t_jit_recv );
	
	llvm_branch( block_in_buffer_exit );
      }
      {
	llvm_set_insert_point(block_not_in_buffer);

	IndexDomainVector args = get_index_vector_from_index( r_multi_index );

	ViewLeaf vv( JitDeviceLayout::LayoutCoalesced , args );
	//ViewLeaf vv( JitDeviceLayout::LayoutCoalesced , r_multi_index );
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
#endif
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
#if 1
      const Map& map = expr.operation().map;
      FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      // int goffsetsId = 
      // void * goffsetsDev = QDPCache::Instance().getDevicePtr( goffsetsId );
      //QDP_info("Map:AddressLeaf: add goffset p=%p",goffsetsDev);
      a.setAddr( const_cast<int*>(expr.operation().map.goffset(a.subset).slice()) );

      void * rcvBuf = NULL;
      if (map.hasOffnode()) {
	const FnMapRsrc& rRSrc = fnmap.getCached();
	rcvBuf = rRSrc.getRecvBufPtr();
      }
      //QDP_info("Map:AddressLeaf: add recv buf p=%p",rcvBufDev);
      a.setAddr(rcvBuf);

      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
#endif
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
    //QDP_error_exit("ni addressleaf map apply");
#if 1
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());

    const int nodeSites = Layout::sitesOnNode();
    int returnVal=0;

    Expr subexpr(expr.child());

    if (map.offnodeP)
      {
#if QDP_DEBUG >= 3
	QDP_info("Map: off-node communications required");
#endif

	//QDPIO::cerr << "map phase 1, off-node required, subset id = " << f.subset.getId() << "\n";

	int dstnum = map.destnodes_num[f.subset.getId()][0]*sizeof(InnerType_t);
	int srcnum = map.srcenodes_num[f.subset.getId()][0]*sizeof(InnerType_t);

	//QDPIO::cerr << "dest source site numbers = " << map.destnodes_num[f.subset.getId()][0] << " " << map.srcenodes_num[f.subset.getId()][0] << "\n";

	const FnMapRsrc& rRSrc = fnmap.getResource(srcnum,dstnum);

	const int my_node = Layout::nodeNumber();

	// Make sure the inner expression's map function
	// send and receive before recursing down
	int maps_involved = forEach(subexpr, f , BitOrCombine());
	if (maps_involved > 0) {
	  QDP_error_exit("shift of shift is not supported");
	  // ShiftPhase2 phase2;
	  // forEach(subexpr, phase2 , NullCombine());
	}

#if 1
	static JitFunction function;

	// Build the function
	if (!function.built())
	  {
	    //std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	    function_gather_build<InnerType_t>( function , rRSrc.getSendBufPtr() , map , subexpr );
	    //std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
	  }
	else
	  {
	    //std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
	  }

	// Execute the function
	function_gather_exec(function, rRSrc.getSendBufPtr() , map , subexpr , f.subset );

	rRSrc.send_receive();
	
	returnVal = maps_involved | map.getId();
#endif
      }
    else 
      {
	returnVal = ForEach<A, ShiftPhase1, BitOrCombine>::apply(expr.child(), f, c);
      }
    return returnVal;
#endif
  }
};




template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase2 , CTag>
{
  //typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t TypeA_t;
  //typedef typename Combine1<TypeA_t, FnMap, OpCombine>::Type_t Type_t;
  //typedef QDPExpr<A,OLattice<Type_t> > Expr;
  typedef int Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const ShiftPhase2 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());
    if (map.offnodeP) {
      const FnMapRsrc& rRSrc = fnmap.getCached();
      //QDP_info("ShiftPhase2: FnMap");

#ifdef JIT_TIMING
      std::vector<double> tt;
      static StopWatch sw;
      sw.start();
#endif

      rRSrc.qmp_wait();

#ifdef JIT_TIMING
      sw.stop();
      tt.push_back( sw.getTimeInMicroseconds() );
      QDPIO::cout << "MPI wait ";
      if (tt.size()>0) {
	for(int i=0 ; i<tt.size() ;++i)
	  QDPIO::cout << tt.at(i) << " ";
	QDPIO::cout << "\n";
      }
#endif

    }
    return ForEach<A, ShiftPhase2, CTag>::apply(expr.child(), f, c);
  }
};




template<class A>
struct ForEach<UnaryNode<FnMap, A>, HasShift , BitOrCombine>
{
  typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t InnerTypeA_t;
  typedef typename Combine1<InnerTypeA_t, FnMap, OpCombine>::Type_t InnerType_t;
  typedef int Type_t;
  typedef QDPExpr<A,OLattice<InnerType_t> > Expr;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const HasShift &f, const BitOrCombine &c)
  {
    return 1;
  }
};



template<class A>
struct ForEach<UnaryNode<FnMap, A>, HasOffNodeShift , BitOrCombine>
{
  typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t InnerTypeA_t;
  typedef typename Combine1<InnerTypeA_t, FnMap, OpCombine>::Type_t InnerType_t;
  typedef int Type_t;
  typedef QDPExpr<A,OLattice<InnerType_t> > Expr;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const HasOffNodeShift &f, const BitOrCombine &c)
  {
    const Map& map = expr.operation().map;

    int ret = 0;

    if (map.offnodeP)
      ret |= map.getId();

    return ret;
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

  const Map& getMap(int isign,int dir) const { return bimapsa((isign+1)>>1,dir); }

private:
  //! Hide copy constructor
  ArrayBiDirectionalMap(const ArrayBiDirectionalMap&) {}

  //! Hide operator=
  void operator=(const ArrayBiDirectionalMap&) {}

private:
  multi2d<Map> bimapsa;
  
};


// Add this code if you need CPU shifts
#if 1
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

    //     if (map.offnodeP) {
    //       if (map.goffsets[f.val1()] < 0) {
    // 	const FnMapRsrc& rRSrc = fnmap.getCached();
    // 	const Type_t *recv_buf_c = rRSrc.getRecvBufPtr<Type_t>();
    // 	Type_t* recv_buf = const_cast<Type_t*>(recv_buf_c);
    // #if QDP_DEBUG >= 3
    // 	if ( recv_buf == 0x0 ) { 
    // 	  QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (recv_buf). Do you use shifts of shifts?"); 
    // 	}
    // #endif
    // 	return recv_buf[-map.goffsets[f.val1()]-1];
    //       } else {

    EvalLeaf1 ff( map.goffsets[all.getId()][f.val1()] );
    return Combine1<TypeA_t, FnMap, CTag>::combine(ForEach<A, EvalLeaf1, CTag>::apply(expr.child(), ff, c),expr.operation(), c);
    //}
  }
};
#endif




} // namespace QDP

#endif
