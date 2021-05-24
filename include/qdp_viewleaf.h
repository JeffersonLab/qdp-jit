#ifndef QDP_VIEWLEAF
#define QDP_VIEWLEAF

namespace QDP {

template<class T>
struct LeafFunctor<QDPTypeJIT<T,OLatticeJIT<T> >, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const QDPTypeJIT<T,OLatticeJIT<T> > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem( v.getLayout() , v.getIndex() ) );
    return reg;
  }
};


template<class T>
struct LeafFunctor<OScalarJIT<T>, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT<T> & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem() );
    return reg;
  }
};



template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< float > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< double > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< int > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};

template<>
struct LeafFunctor<OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >, ViewLeaf>
{
  typedef typename REGType< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > >::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT< PScalarJIT < RScalarJIT< WordJIT< bool > > > > > & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup_value( s.elem() );
    return reg;
  }
};

  
  

template<class T>
struct LeafFunctor<OLatticeJIT<T>, ViewLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLatticeJIT<T> & s, const ViewLeaf& v)
  {
    Type_t reg;
    reg.setup( s.elem( v.getLayout() , v.getIndex() ) );
    return reg;
  }
};





template<class T>
struct LeafFunctor<QDPType<T,OLattice<T> >, JitCreateLoopsLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const QDPType<T,OLattice<T> > & s, const JitCreateLoopsLeaf& v)
  {
    return 0;
  }
};

template<class T>
struct LeafFunctor<QDPType<T,OScalar<T> >, JitCreateLoopsLeaf>
{
  typedef int Type_t;
  inline static
  Type_t apply(const QDPType<T,OScalar<T> > & s, const JitCreateLoopsLeaf& v)
  {
    return 0;
  }
};


template<class T>
struct LeafFunctor<OLattice<T>, JitCreateLoopsLeaf >
{
  typedef int Type_t;
  inline static
  Type_t apply(const OLattice<T> & s, const JitCreateLoopsLeaf& v)
  {
    QDPIO::cout << "create generic\n";
    return 0;
  }
};

  template<class T>
struct LeafFunctor<OScalar<T>, JitCreateLoopsLeaf >
{
  typedef int Type_t;
  inline static
  Type_t apply(const OScalar<T> & s, const JitCreateLoopsLeaf& v)
  {
    return 0;
  }
};

  
  template<class T, int N>
struct LeafFunctor<OLattice<PSpinMatrix<T,N> >, JitCreateLoopsLeaf >
{
  typedef int Type_t;
  inline static
  Type_t apply(const OLattice<PSpinMatrix<T,N> > & s, const JitCreateLoopsLeaf& v)
  {
    if ( ! v.ops_only )
      {
	QDPIO::cout << "create spin mat\n";
	
	QDPIO::cout << "loop(0,N)\n";
	//v.loops.push_back( JitForLoop(0,N) );
	v.loop_bounds.push_back( N );

	QDPIO::cout << "loop(0,N)\n";
	//v.loops.push_back( JitForLoop(0,N) );
	v.loop_bounds.push_back( N );
      }
    return 0;
  }
};



  

  

template<class T>
struct LeafFunctor<OLatticeJIT<PScalarJIT<T> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLatticeJIT<PScalarJIT<T> > & s, const ViewSpinLeaf& v)
  {
    return s.elem( v.getLayout() , v.getIndex() ).getRegElem();
  }
};


template<class T,int N>
struct LeafFunctor<OLatticeJIT<PSpinMatrixJIT<T,N> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OLatticeJIT<PSpinMatrixJIT<T,N> > & s, const ViewSpinLeaf& v)
  {
    if (v.ind.size() != 2)
      {
	QDPIO::cout << "at spinmat but not 2 indices provided (" << v.ind.size() << ")" << std::endl;
	QDP_abort(1);
      }

    QDPIO::cout << "leaf spinmat " << v.ind[0] << " " << v.ind[1] << std::endl;
    
    return s.elem( v.getLayout() , v.getIndex() ).getRegElem( v.loops[ v.ind[0] ].index() , v.loops[ v.ind[1] ].index() );
  }
};

  
  // rv.right().right().elem( JitDeviceLayout::Coalesced , r_idx ).getRegElem( loop_e.index() , loop_c.index() );

template<class T>
struct LeafFunctor<OScalarJIT< PScalarJIT<T> >, ViewSpinLeaf>
{
  typedef typename REGType<T>::Type_t Type_t;
  inline static
  Type_t apply(const OScalarJIT< PScalarJIT<T> > & s, const ViewSpinLeaf& v)
  {
    return s.elem().getRegElem();
  }
};





/// JIT2BASE

template<class T>
struct LeafFunctor<OScalarJIT<T>, JIT2BASE>
{
  typedef OScalar< typename BASEType<T>::Type_t > Type_t;
  inline static
  Type_t apply(const OScalarJIT<T> & s, const JIT2BASE& v)
  {
    Type_t r;
    return r;
  }
};

template<class T>
struct LeafFunctor<OLatticeJIT<T>, JIT2BASE>
{
  typedef OLattice< typename BASEType<T>::Type_t >  Type_t;
  //typedef int Type_t;
  inline static
  Type_t apply(const OLatticeJIT<T> & s, const JIT2BASE& v)
  {
    Type_t r;
    return r;
  }
};



}

#endif
