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
  const multi1d<int>& goffset() const {return goffsets;}
  const multi1d<int>& soffset() const {return soffsets;}
  const multi1d<int>& roffset() const {return roffsets;}
  int getRoffsetsId() const { return roffsetsId;}
  int getSoffsetsId() const { return soffsetsId;}
  int getGoffsetsId() const { return goffsetsId;}

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
  multi1d<int> goffsets;
  multi1d<int> soffsets;
  multi1d<int> srcnode;
  multi1d<int> dstnode;

  multi1d<int> roffsets;

  int roffsetsId;
  int soffsetsId;
  int goffsetsId;
  int myId; // master map id

  multi1d<int> srcenodes;
  multi1d<int> destnodes;

  multi1d<int> srcenodes_num;
  multi1d<int> destnodes_num;

  // Indicate off-node communications is needed;
  bool offnodeP;
};




struct FnMap
{
  //PETE_EMPTY_CONSTRUCTORS(FnMap)
private:
  FnMap& operator=(const FnMap& f);

public:
  const Map& map;
  //std::shared_ptr<RsrcWrapper> pRsrc;
  QDPHandle::Handle<RsrcWrapper> pRsrc;

  FnMap(const Map& m): map(m), pRsrc(new RsrcWrapper( m.destnodes , m.srcenodes )) {}
  FnMap(const FnMap& f) : map(f.map) , pRsrc(f.pRsrc) {}

  const FnMapRsrc& getResource(int srcnum_, int dstnum_) {
    return (*pRsrc).getResource( srcnum_ , dstnum_ );
  }

  const FnMapRsrc& getCached() const {
    return (*pRsrc).get();
  }
  
  template<class T>
  inline typename UnaryReturn<T, FnMap>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }

};


struct FnMapJIT 
{
public:
  Jit& jit;
  Jit::IndexRet index;
  const Map& map;
  QDPHandle::Handle<RsrcWrapper> pRsrc;

  FnMapJIT(const FnMap& fnmap,const Jit::IndexRet& i,Jit& j): map(fnmap.map), pRsrc(fnmap.pRsrc), index(i), jit(j) {}
  FnMapJIT(const FnMapJIT& f) : map(f.map) , pRsrc(f.pRsrc), index(f.index), jit(f.jit) {}

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



#if 1
    // typedef ForEachTypeParser<A, ParseTag,ParseTag, CTag> ForEachA_t;
    // typedef OpVisitor<FnMap, ParseTag>          Visitor_t;
    // typedef typename ForEachA_t::Type_t   TypeA_t;
    // typedef Combine1<TypeA_t, FnMap, CTag>   Combiner_t;
    // typedef typename Combiner_t::Type_t   Type_t;
    // typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t AInnerTypeA_t;
    // typedef typename Combine1<AInnerTypeA_t, FnMap, OpCombine>::Type_t InnerTypeBB_t;




#if 1
template<class A>
struct ForEach<UnaryNode<FnMap, A>, ParamLeaf, TreeCombine>
  {
    typedef typename ForEach< UnaryNode<FnMapJIT, A> , ParamLeaf, TreeCombine>::Type_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnMap, A>& expr, const ParamLeaf &p, const TreeCombine &c)
    {
      const Map& map = expr.operation().map;
      FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      Jit::IndexRet index = p.getFunc().addParamIndexFieldRcvBuf();

      ParamLeaf pp( p.getFunc() , index.r_newidx );

      return Type_t( FnMapJIT( expr.operation() , index , p.getFunc() ) , 
		     ForEach< A, ParamLeaf, TreeCombine >::apply( expr.child() , pp , c ) );

      //ForEach< UnaryNode<FnMapJIT, A>, ParamLeaf, TreeCombine>::apply(expr.child() , pp , c );


      // TypeA_t A_val  = ForEachA_t::apply(expr.child(), pp, t);
      // Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      //ParseTag ff( f.getJitArgs() , newIdx.str() );
      //TypeA_t A_val  = ForEachA_t::apply(expr.child(), ff, ff, c);
      //Type_t val = Combiner_t::combine(A_val, expr.operation(), c);
      //f.ossCode << ff.ossCode.str();

      //return val;
    }
  };

#else

template<class A>
struct ForEach<UnaryNode<FnMap, A>, ParamLeaf, TreeCombine>
  {
    typedef typename ForEach< A , ParamLeaf, TreeCombine>::Type_t TypeA_t;
    typedef TypeA_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnMap, A>& expr, const ParamLeaf &p, const TreeCombine &c)
    {
      const Map& map = expr.operation().map;
      FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      // This moves to AddressLeaf Functor !
      //
      // int goffsetsId = expr.operation().map.getGoffsetsId();
      // void * goffsetsDev = QDPCache::Instance().getDevicePtr( goffsetsId );
      // int posGoff = f.getJitArgs().addPtr( goffsetsDev );


      // get receive buffer on device and NULL otherwise
      //

      Jit::IndexRet index = p.getFunc().addParamIndexFieldRcvBuf();

#if 0
#endif


      // string codeTypeA;
      // typedef InnerTypeBB_t TTT;
      // TTT ttt;
      // getTypeStringT<TTT>( codeTypeA , f.getJitArgs() );

      // ostringstream newIdx;
      // newIdx << "((int*)(" << f.getJitArgs().getPtrName() << "[ " << posGoff  << " ].ptr))" << "[" << f.getIndex() << "]";

      ParamLeaf pp( p.getFunc() , index.r_newidx );
      return Type_t( ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child() , pp , c ) );

      // TypeA_t A_val  = ForEachA_t::apply(expr.child(), pp, t);
      // Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

#if 0
      ostringstream code;
      code << newIdx.str() << " < 0 ? " <<
	"(" <<
	"((" << codeTypeA << "*)(" << f.getJitArgs().getPtrName() << 
	"[ " << posRcvBuf  << " ].ptr))" << "[-" << newIdx.str() << "-1]" << 
	"):(" <<
	ff.ossCode.str() <<
	")";

      f.ossCode << code.str();
#endif

      //ParseTag ff( f.getJitArgs() , newIdx.str() );
      //TypeA_t A_val  = ForEachA_t::apply(expr.child(), ff, ff, c);
      //Type_t val = Combiner_t::combine(A_val, expr.operation(), c);
      //f.ossCode << ff.ossCode.str();

      //return val;
    }
  };

#endif




template<class A>
struct ForEach<UnaryNode<FnMapJIT, A>, ViewLeaf, OpCombine>
  {
    //typedef typename ForEach< UnaryNode<FnMapJIT, A> , ParamLeaf, TreeCombine>::Type_t Type_t;
    typedef typename ForEach<A, ViewLeaf, OpCombine>::Type_t TypeA_t;
    typedef typename Combine1<TypeA_t, FnMapJIT , OpCombine>::Type_t Type_t;
    inline
    static Type_t apply(const UnaryNode<FnMapJIT, A>& expr, const ViewLeaf &v, const OpCombine &o)
    {
      //const Map& map = expr.operation().map;
      //FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      Jit::IndexRet index = expr.operation().index;

      Jit& func = expr.operation().jit;
      
      Type_t ret(func);
      func.addCondBranch(index);

      ret = Combine1<TypeA_t, FnMapJIT , OpCombine>::combine(ForEach<A, ViewLeaf, OpCombine>::apply(expr.child(), v, o),
								    expr.operation(), o);

      func.addCondBranch2();
      printme<Type_t>("FnMapJIT ViewLeaf");
      Type_t recv_buf(func);
      ret = recv_buf;

      return ret;

      // return Type_t( FnMapJIT(expr.operation(),index) , ForEach<A, ParamLeaf, TreeCombine>::apply(expr.child() , pp , c ) );

      //ForEach< UnaryNode<FnMapJIT, A>, ParamLeaf, TreeCombine>::apply(expr.child() , pp , c );


      // TypeA_t A_val  = ForEachA_t::apply(expr.child(), pp, t);
      // Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      //ParseTag ff( f.getJitArgs() , newIdx.str() );
      //TypeA_t A_val  = ForEachA_t::apply(expr.child(), ff, ff, c);
      //Type_t val = Combiner_t::combine(A_val, expr.operation(), c);
      //f.ossCode << ff.ossCode.str();

      //return val;
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

      //QDP_info("ForEach 0");
      int goffsetsId = expr.operation().map.getGoffsetsId();
      //QDP_info("ForEach 1 %d",goffsetsId);
      void * goffsetsDev = QDPCache::Instance().getDevicePtr( goffsetsId );
      //QDP_info("ForEach 2 %p",goffsetsDev);
      QDP_info("Map:AddressLeaf: add goffset p=%p",goffsetsDev);
      a.setAddr(goffsetsDev);
      //QDP_info("ForEach 3");

      void * rcvBufDev;
      if (map.hasOffnode()) {
	const FnMapRsrc& rRSrc = fnmap.getCached();
	int rcvId = rRSrc.getRcvId();
	rcvBufDev = QDPCache::Instance().getDevicePtr( rcvId );
      } else {
	rcvBufDev = NULL;
      }
      QDP_info("Map:AddressLeaf: add recv buf p=%p",rcvBufDev);
      a.setAddr(rcvBufDev);


      return Type_t( ForEach<A, AddressLeaf, NullCombine>::apply( expr.child() , a , n ) );
    }
  };



#endif




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

    const int nodeSites = Layout::sitesOnNode();
    int returnVal=0;

    Expr subexpr(expr.child());

    if (map.offnodeP)
      {
#if QDP_DEBUG >= 3
#endif
	QDP_info("Map: off-node communications required");

	int dstnum = map.destnodes_num[0]*sizeof(InnerType_t);
	int srcnum = map.srcenodes_num[0]*sizeof(InnerType_t);

	const FnMapRsrc& rRSrc = fnmap.getResource(srcnum,dstnum);

	const int my_node = Layout::nodeNumber();

	// Make sure the inner expression's map function
	// send and receive before recursing down
	int maps_involved = forEach(subexpr, f , BitOrCombine());
	if (maps_involved > 0) {
	  ShiftPhase2 phase2;
	  forEach(subexpr, phase2 , NullCombine());
	}

#if 1
	// map.soffsets.size();
	// QDPCache::Instance().getDevicePtr( map.getSoffsetsId() );
	// typename InnerType_t; // type on 1 site
	// subexpr; //
	// rRSrc.getSendBufDevPtr(); // send buffer

	// Gather subexpr evaluated on soffset[] into send buffer
	static CUfunction function;

	// Build the function
	if (function == NULL)
	  {
	    std::cout << __PRETTY_FUNCTION__ << ": does not exist - will build\n";
	    function = function_gather_build<InnerType_t>( rRSrc.getSendBufDevPtr() , map , subexpr );
	    std::cout << __PRETTY_FUNCTION__ << ": did not exist - finished building\n";
	  }
	else
	  {
	    std::cout << __PRETTY_FUNCTION__ << ": is already built\n";
	  }

	// Execute the function
	function_gather_exec(function, rRSrc.getSendBufDevPtr() , map , subexpr );

	rRSrc.send_receive();
	
	returnVal = maps_involved | map.getId();
#endif
      } 
    else 
      {
	QDP_info("Map: no off-node comms");
	returnVal = ForEach<A, ShiftPhase1, BitOrCombine>::apply(expr.child(), f, c);
      }
    return returnVal;
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
      rRSrc.qmp_wait();
    }
    ForEach<A, ShiftPhase2, CTag>::apply(expr.child(), f, c);
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
